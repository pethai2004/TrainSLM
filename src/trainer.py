# Full training script 
import os 
import logging 
import subprocess
import logging
import tqdm
from typing import Dict, List, Any, Union, Tuple
from contextlib import contextmanager, ExitStack, nullcontext

import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from datasets import Dataset
import transformers as tfm

from config import TrainingConfig, TrainingState
from utility import * 
from io_utility import *
from dist import *
from dist import _put
from data_utility import *

logger = logging.getLogger(__name__)

class Trainer:
    '''Main trainer class for training the model.'''
    
    def __init__(
        self, 
        training_config: TrainingConfig, 
        model: nn.Module, 
        tokenizer: tfm.PreTrainedTokenizerFast,
        **kwargs
    ):
        """Available kwargs: `output_dir`, `optimizer`, `scheduler`, `train_dataset`, `eval_dataset`"""
        training_config._post_init()
        self.config = training_config
        self.model = model
        self.tokenizer = tokenizer
        if kwargs.get("output_dir", None) is not None:
            if self.config.output_dir is not None:
                self.config.output_dir = output_dir
                log_on_main(f"Overriding config.output_dir with {output_dir}")
        else:
            output_dir = self.config.output_dir
        self.output_dir = output_dir
        self.output_trial_dir = os.path.join(output_dir, self.config.trial_name)
        self._acc_FLOPS = 0
        self._curr_FLOPS = 0
        self._activated_neftune = None
        self._is_model_initialized = False
        self._past = None 
        self._should_skip_update = False # indicate that the current loss is NaN or Inf
        self._should_log_flops = False
        self._model_num_params = None 
        self._step = 0  
        self._collate_fn = collate_fn
        self.should_sync = False
        self.train_dataset = kwargs.get("train_dataset", None)
        self.eval_dataset = kwargs.get("eval_dataset", None)
        self.optimizer = kwargs.get("optimizer", None)
        self.scheduler = kwargs.get("scheduler", None)
        self._train_dataset_iter = None
        self._eval_dataset_iter = None
        
        os.makedirs(self.output_trial_dir, exist_ok=True)
        self.state = TrainingState()
        # will always not raise regardless of the availability of the flash_attn_2
        self._set_seed()
        torch.backends.cuda.enable_flash_sdp(enabled=True) #(always default to flash_attn_2)
        if tfm.modeling_utils.is_flash_attn_2_available():
            self.config.attn_implementation = "flash_attention_2"
        else: 
            torch.backends.cuda.enable_mem_efficient_sdp(enabled=True) 
        self.gradient_scaler = None 
        if self.config.precision in ["fp16", "bfp32"]:
            self.gradient_scaler = torch.amp.GradScaler() # maybe just use enabled=False
        if self.model is not None:
            self.model = self._model_post_init(self.model)
            self._is_model_initialized = True
        self._init_tensorboard()
        if self.config.token is not None:
            self._create_hgf_repos()
        if tokenizer is None: 
            if self.config.tokenizer_name_or_path is not None:
                self.tokenizer = get_tokenizer(self.config.tokenizer_path)
        self.tb_log_dir = None # will be set later
        
    @property
    def _device(self):
        '''Get the device: `device:rank`'''
        return f"{self.config.device}:{D.get_rank()}"
    
    def _set_seed(self):
        """Set seed for the trainer."""
        self.seed = self.config.seed if not self.config.seed_for_each_worker else self.config.seed + D.get_rank()
        set_seed(self.seed, self.config.full_determinism)
        
    def _create_hgf_repos(self):
        """Create HuggingFace repos."""
        if D.get_rank() == 0:
            assert self.config.token is not None, "Token must be provided to create HuggingFace repos."
            if self.config.push_to_hub_model_id is None:
                self.config.push_to_hub_model_id = f"{self.output_dir}_{self.config.trial_name}"
            from huggingface_hub import HfApi
            api = HfApi()
            self.repo_id = api.create_repo(self.config.push_to_hub_model_id, exist_ok=True, token=self.config.token).repo_id
            with open(os.path.join(self.output_trial_dir, ".gitignore"), "w+") as gitignore:
                if "global_training_steps*" not in gitignore:
                    gitignore.write("global_training_steps*\n")
                if "global_epochs*" not in gitignore:
                    gitignore.write("global_epochs*\n")
        D.barrier()
        
    def _model_post_init(self, model):
        
        assert self.tokenizer is not None, "Tokenizer must be initialized before calling `_model_post_init`."
        if hasattr(model, "get_input_embeddings"):
            if not model.get_input_embeddings() == len(self.tokenizer):
                model.resize_token_embeddings(len(self.tokenizer))
                log_on_main(f"Resized model token embeddings to {len(self.tokenizer)}")
        else:
            log_on_main(f"Model does not have `get_input_embeddings` method, skipping resizing token embeddings. This may lead to unexpected behavior.")
        if self.config.gradient_checkpointing is not None:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            else: 
                log_on_main("Model may not support gradient checkpointing and does not have `gradient_checkpointing_enable` method.")
        
        if self.config.neftune_noise_alpha > 0.0:
            model = activate_neftune(self, model)
            
        if isinstance(model, nn.Module):
            self._model_num_params = model.num_parameters()
            device = self._device
            model.to(device=device, dtype=self.config.precision)
            device_ids = None if not cuda.is_available() else [device]
            model = DDP(model, device_ids=device_ids, output_device=device) # just set output_device to the current device (TODO: add Parallel)
        return model
            
    def _init_tensorboard(self, dir_path: str=None):
        self.tb_log_dir = os.path.join(os.getcwd(), self.output_trial_dir, tensorboard_log_dir) if dir_path is None else dir_path
        self.writer = SummaryWriter(log_dir=self.tb_log_dir)
        log_on_main(f"Tensorboard initialized at {self.tb_log_dir}")

    def train(self, resume_from_checkpoint=True):
        """Main training loop method. If `resume_from_checkpoint` is True, 
            it will try to find potential checkpointing, or can be passed directly as path. """
        prepare_start = time.time()
        arg = self.config
        self.state.is_in_train = True
        self.state.trial_name = arg.trial_name
        self._set_seed() # set again
        
        train_dataset, eval_dataset = self.get_dataset().values()
        should_eval = True if self.eval_dataset is not None else False
        
        if self.optimizer is None or self.scheduler is None:
            self.optimizer, self.scheduler = self.create_optimizer_and_scheduler()
            log_on_main(f"Trainer initalized optimizer and scheduler: {self.optimizer.__class__.__name__}, {self.scheduler.__class__.__name__}")
            
        reload_model = False 
        if resume_from_checkpoint is not None:
            latest_checkpoint = None 
            if isinstance(resume_from_checkpoint, str) and validate_checkpoint(resume_from_checkpoint):
                latest_checkpoint = resume_from_checkpoint
            else: # resume_from_checkpoint is True
                latest_checkpoint = get_latest_checkpoint(self.output_trial_dir)
            if latest_checkpoint is not None:
                load_checkpoint(
                    from_safe_tensor=arg.save_safe_tensor,
                    checkpoint_path=latest_checkpoint,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    training_config=arg,
                    training_state=self.state
                )
                self.reload_model = True
                self._is_model_initialized = True
            else:
                log_on_main("No checkpoint found, training from scratch.")
        
        cuda.empty_cache()      
        self.dataset_length = {"train": len(train_dataset), "eval": len(self.eval_dataset)}
    
        if reload_model:
            self.model = self._model_post_init(self.model)
        # at this point `model` should be DistributedDataParallel already
        grad_step = self.config.gradient_accumulation_steps
        total_batch_size = arg.full_batch_size 
        num_epochs = arg.num_epochs
        train_length = self.dataset_length["train"] # in batch already train_length = full_batch_size * num_train_example
        num_update_per_epoch = max(train_length // grad_step, 1)
    
        if arg.max_training_steps > 0: # `-1` means no limit, so we don't need to do anything
            if arg.max_training_steps <= num_update_per_epoch:
                old = num_update_per_epoch
                num_update_per_epoch = min(num_update_per_epoch, arg.max_training_steps) 
                train_length = num_update_per_epoch * grad_step # will not truncate any dataset, will reference to `global_training_step` instead
                # note: `train_length` here may no reflect the actual number of training samples as we recalculate the `num_update_per_epoch`
                log_on_main(f"Truncated training length from {old} to {num_update_per_epoch} * {grad_step} = {train_length}")
        actual_training_samples = num_update_per_epoch * grad_step # per epoch
        total_training_steps = num_epochs * num_update_per_epoch
        starting_epoch = self.state.global_epoch

        starting_step = self.state.num_batch_training_so_far % actual_training_samples # is a remainder, ignore the epoch
        num_batch_to_skip = starting_step
        #sanity check
        if num_batch_to_skip + (starting_epoch * actual_training_samples) != self.state.num_batch_training_so_far:
            log_on_main(f"Miscalculation: num_batch_to_skip + (starting_epoch * actual_training_samples) != self.state.num_batch_training_so_far: \
                            {num_batch_to_skip} + ({starting_epoch} * {actual_training_samples}) != {self.state.num_batch_training_so_far}")
        if not isinstance(self.model.module, nn.Module):
            raise ValueError(f"It seems that model is being wrapped multiple times by DistributedDataParallel. Please check the model initialization.")

        num_params = self._model_num_params if self._model_num_params is not None else self.model.module.num_parameters()
        self._steps = self.state.num_batch_training_so_far
        print_configs = {
            "num_parameters": num_params,
            "per_device_batch_size": arg.per_device_batch_size,
            "gradient_accumulation_steps": arg.gradient_accumulation_steps,
            "full_batch_size": arg.full_batch_size,
            "num_processes": D.get_world_size(), # or arg.num_processes
            "num_epochs": f"{starting_epoch} / {num_epochs}",
            "total_grad_step_so_far": f"{self.state.global_training_step} / {total_training_steps}",
            "total_batch_step_so_far" : f"{self.state.num_batch_training_so_far} / {actual_training_samples * num_epochs}",
        }
        print_configs.update(arg.print_option) # combine with other config
        for i, v in print_configs.items(): 
            log_on_main(f"|| \033[1m\033[1;31m{i}\033[0m: \033[0;30m{v}\033[0m") 
        D.barrier()
        prepare_end = time.time() - prepare_start
        log_on_main(f"Preparation took {prepare_end:.2f} seconds")
        log_on_main(f"\033[1m\033[32m--------------------------------------------->> START TRAINING <<-----------------------------------------------\033[0m")
        
        on_trace = False if self.tb_log_dir is None else torch.profiler.tensorboard_trace_handler(self.tb_log_dir)
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=3, warmup=2, active=5, repeat=5),
            on_trace_ready=on_trace,
            profile_memory=True,
            with_flops=True,
            with_modules=True,
            use_cuda=cuda.is_available(),
            with_stack=True
        ) as prof:
            
            for epoch in range(starting_epoch, num_epochs):
                
                self.state.global_epoch = epoch
                epoch_start_time = time.time()
                self.model.train()
                self.model.zero_grad()
                self.optimizer.zero_grad()
                
                if hasattr(self.train_dataset, "set_epoch"):
                    self.train_dataset.set_epoch(epoch)
                
                for step, inputs in enumerate(self.train_dataset):
                    while num_batch_to_skip > 0:
                        num_batch_to_skip -= 1
                        if num_batch_to_skip == 0:
                            logger.warning(f"Skipped {num_batch_to_skip} batches")
                        continue
                        
                    loss = self.inner_loop(inputs=inputs)       
                    self.state.num_batch_training_so_far += 1

                    if self.state.num_batch_training_so_far % arg.log_interval == 0:
                        if D.get_rank() == 0 and D.get_rank() == 0:
                            loss = loss / arg.log_interval
                            lr = self.scheduler.get_last_lr()[0]
                            message = f"EPOCH: {epoch}, GLOBAL_TRAINING_STEPS: {self.state.global_training_step}, BATCH_STEPS: {step}, LOSS: {loss:.4f}, LR: {lr:.4f}" 
                            logger.warning(message)
                            self.writer.add_scalar("train/loss", loss, self.state.global_training_step)
                            self.writer.add_scalar("train/lr", lr, self.state.global_training_step)
                        D.barrier()
                        
                    if self.state.num_batch_training_so_far % arg.checkpoint_interval == 0:
                        if D.get_rank() == 0:
                            save_checkpoint(
                                checkpoint_dir=self.output_trial_dir,
                                model=self.model, 
                                optimizer=self.optimizer, 
                                scheduler=self.scheduler, 
                                training_config=arg, 
                                training_state=self.state, 
                            )
                            logger.warning(f"Checkpoint saved at global_training_step: {self.state.global_training_step}")
                        D.barrier()
                    
                    prof.step() # we will profile every step (batch)
                self.state.global_epoch += 1
                epoch_end_time = time.time()
                epoch_time = epoch_end_time - epoch_start_time
                logger.warning(f"Epoch {epoch} took {epoch_time:.2f} seconds")
                if self.state.global_training_step >= total_training_steps:
                    logger.warning(f"Training completed at global_training_step: {self.state.global_training_step}")
                    break
            self.state.is_in_train = False  
            D.barrier()
            log_on_main(f"\033[1m\033[32m--------------------------------------------->> END TRAINING <<-----------------------------------------------\033[0m")
        
    def sample(self, inputs, *args, **kwargs):
        """Override this method for custome behavior for how the inputs should be sampled."""
        if not isinstance(inputs, dict):
            raise ValueError(f"Sample must be a dictionary, got: {type(inputs)}")
        if len(inputs) == 0:
            raise ValueError("Empty sample provided.")
        if self.config.dispatch_on_device:
            inputs = _put(inputs, device=self._device, non_blocking=self.config.non_blocking)
        if not "labels" in inputs: # assume this is causal generation task
            try:
                labels = inputs["input_ids"].clone() 
            except AttributeError:
                labels = inputs["input_ids"] # just a list
            inputs["labels"] = labels
        # if self.config.model_input_name not in inputs: ######## TODO
        #     raise ValueError(f"Key {self.config.model_input_name} not found in inputs, found keys: {list(sample.keys())}")
        if not self.state.is_in_train and self.config.past_input_name is not None \
            and self.config.past_input_name in inputs: # in eval mode and past is provided
            self._past = inputs[self.config.past_input_name]
            
        return inputs
    
    @contextlib.contextmanager
    def accumulate(self, model, force=False):
        """Auto accumulate the gradients."""
        grad_step = self.config.gradient_accumulation_steps
        if force: 
            grad_step = 1
        self._steps += 1
        self.should_sync = self._steps % grad_step == 0  
        
        with contextlib.ExitStack() as stack:
            if not self.should_sync: # if not self.should_sync then return will yield null context else yield mode.no_sync 
                stack.enter_context(getattr(model, "no_sync")())        
            else:
                stack.enter_context(contextlib.nullcontext())
            yield 
    
    def step(self, loss):
        '''Stepping optimizer and scheduler. Wiill do necessary scaling here'''
        if self.gradient_scaler is not None:
            self.gradient_scaler.scale(loss).backward()
        loss = loss.detach()
        if self.should_sync: # gradient synchronous steps
            if self.gradient_scaler is not None:
                
                self.gradient_scaler.unscale_(self.optimizer)# explicitly call upscale for clipping
                if self.config.gradient_clip_norm > 0.:
                    torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), self.config.gradient_clip_norm)
                if self.config.gradient_clip_value > 0.:
                    torch.nn.utils.clip_grad_value_(self.model.module.parameters(), self.config.gradient_clip_value)
                
                self.gradient_scaler.step(self.optimizer) # will not internally call `unscale_` again
                self.gradient_scaler.update() 
            else: 
                self.optimizer.step()
                
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.state.global_training_step += 1
        
        return loss 
    
    def inner_loop(self, inputs, *args, **kwargs):
        """Inner training loop, reponsible for forward, backwarn pass, optimizer and lr step, and gradient accumulation."""
        
        with self.accumulate(self.model):
            with torch.autocast(enabled=self.config.mixed_precision != "off", cache_enabled=False):
                loss = self.compute_loss(inputs)
                loss = loss / self.config.gradient_accumulation_steps
        
        loss = self.step(loss)
        self.state.loss = loss
        if self.config.report_nan:
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"Loss is NaN or Inf at epoch {self.state.global_epoch}, step {self.state.global_training_step}, loss: {loss}")
                self._should_skip_update = True
        
        if hasattr(self.model, "floating_point_ops") and self._should_log_flops:
            self._curr_FLOPS = self.model.floating_point_ops(inputs)
            self._acc_FLOPS += self._curr_FLOPS
            
        return loss

    def compute_loss(self, inputs, *args, **kwargs):
        """Override this for custome behavior"""
        inputs = self.sample(inputs, *args, **kwargs)
        output = self.model(**inputs)
        loss = output.loss
        if loss is None:
            raise ValueError(f"Loss is None, please make sure the model return loss.")
        if D.get_world_size() > 1:
            loss = loss.mean()
        return loss
    
    def evaluate(self, dataset=None) -> Dict[str, Any]:
        '''Evaluate the model.'''
        if self.eval_dataset is None and dataset is None:
            log_on_main(f"No eval dataset provided, no evaluation will be performed.")
            return {}
        start_time = time.time()
        self.state.is_in_train = False
        self.model.eval()
        eval_loss = 0.0
        num_eval_steps = 0
        eval_dataset = self.eval_dataset if dataset is None else dataset
        
        for inputs in eval_dataset:
            with torch.no_grad():
                loss = self.compute_loss(inputs)
                eval_loss += loss
                num_eval_steps += 1
            
        eval_loss = eval_loss / num_eval_steps
        eval_time = time.time() - start_time
        log_on_main(f"Evaluation loss: {eval_loss:.4f}, time: {eval_time:.2f}")
        
    def get_one_training_sample(self, *args, **kwargs):
        '''Get one training batch sample.'''
        
    def create_prepared_dataset(self, dataset: Dataset, group=True, *args, **kwargs) -> DataLoader:
        """Create the prepared dataset."""
        if not hasattr(dataset, "__len__"):
            raise NotImplementedError("Unsupported dataset does not implement `__len__` methods. For dataset of type `DatasetDict` or iterator will be added later")

        if self.config.max_training_examples > 0 and len(dataset) > self.config.max_training_examples:
            dataset = dataset.select(range(self.config.max_training_examples))
            log_on_main(f"Truncated dataset to `self.config.max_training_examples` = {self.config.max_training_examples} examples from {len(dataset)} examples")

        if group:
            sampler = DistributedGroupedSampler(
                dataset, 
                num_multi_batch=None, 
                model_input_name=self.config.model_input_name[0], 
                shuffle=False, # always False
                seed=self.config.seed, 
                drop_last=True
            ) 
        else:
            sampler = DistributedSampler(
                dataset, 
                num_replicas=D.get_world_size(), # or self.config.num_processes
                rank=D.get_rank(), 
                shuffle=False, 
                seed=self.config.seed, 
                drop_last=True
            )
        if self.config.pin_memory:
            if cuda.is_available(): # althrough DataLoader default to cuda if pin_memory_device is None, we explicitly set it here
                pin_memory_device = self._device
            else:
                self.config.pin_memory = False
                pin_memory_device = None
                logger.warning(f"Auto disabled pin_memory")
                
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config.per_device_batch_size,
            sampler=sampler,
            collate_fn=self._collate_fn, 
            num_workers=self.config.num_dataloader_workers, 
            pin_memory=self.config.pin_memory,
            pin_memory_device=pin_memory_device,
            drop_last=True,
            persistent_workers=self.config.persistent_workers
        )
        return dataloader
    
    def split_train_eval(self, dataset):

        if isinstance(self.config.train_test_split, int):
            train_length = min(self.config.train_test_split / len(dataset), 1) # if 1, then no eval dataset
            
        train_length = max(1 - self.config.train_test_split, self.config.train_test_split) * len(dataset)
        train_length = int((train_length // self.config.full_batch_size) * self.config.full_batch_size) # round down to the nearest batch size
        eval_length = int(len(dataset) - train_length)
        train_dataset, eval_dataset = dataset.train_test_split(train_size=train_length, test_size=eval_length).values()
        
        return train_dataset, eval_dataset
    
    def create_training_dataset(self, dataset, *args, **kwargs) -> DataLoader:
        """Create the training dataset. Either override and then call `super().create_training_dataset` or provide a dataset in kwargs. Override this method for custom behavior."""

        if not isinstance(dataset, Dataset):
            raise ValueError(f"Dataset must be of type `Dataset`, got: {type(dataset)}")
        return self.create_prepared_dataset(dataset, *args, **kwargs)
    
    def create_eval_dataset(self, dataset=None, *args, **kwargs) -> DataLoader:
        """Create the evaluation dataset. Either override and then call `super().create_eval_dataset` or provide a dataset in kwargs. Override this method for custom behavior."""
        if dataset is None:
            log_on_main(f"No eval dataset provided, no evaluation will be performed.")
            return None
        if not isinstance(dataset, Dataset):
            raise ValueError(f"Dataset must be of type `Dataset`, got: {type(dataset)}")
        return self.create_prepared_dataset(dataset, *args, **kwargs)
        
    def model_initialize(self, *args, **kwargs):
        """Override this method for custom model initialization."""
        return None
    
    def create_optimizer_and_scheduler(self):
        """Create optimizer and scheduler. Override this method for custom optimizer and scheduler."""
        if not self._is_model_initialized or self.model is None:
            raise ValueError(f"Can not initialize optimizer and scheduler without model being initialized.")
        return create_optimizer_and_lr(self, self.config.optimizer_kwargs, self.config.scheduler_kwargs)
    
