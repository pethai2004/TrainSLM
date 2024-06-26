 #TODO: make sharded load and save (transformers.modeling_utils.load_sharded_checkpoint)
 # TODO: add load and save checkpoint for Hub optional if specify config.model_push_to_hub_id / config.token
 # so I need to modify all of the function related to load and save modeling uility
 # TODO: ensure model saved from DDP can be loaded back to DDP and vice versa
 #TODO: delete accelerate
import os 
import re 
import time
import random
import shutil
import pickle
import datetime
from typing import Iterable

import torch
from torch import cuda
import numpy as np

from huggingface_hub import HfApi
from huggingface_hub import get_hf_file_metadata, hf_hub_url, repo_info
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError

from accelerate.utils import save, is_xpu_available, is_torch_xla_available
from accelerate.state import PartialState, DistributedType
from accelerate.data_loader import IterableDatasetShard, SeedableRandomSampler

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm # type: ignore
    
from src.config import *

def validate_checkpoint(checkpoint_path: str) -> bool:
    """
    Validate the checkpoint by checking the existence of the model, optimizer, scheduler, training state, and training config.
    """
    return all([
        os.path.exists(os.path.join(checkpoint_path, MODEL_PATH)),
        os.path.exists(os.path.join(checkpoint_path, OPTIMIZER_PATH)),
        os.path.exists(os.path.join(checkpoint_path, SCHEDULE_PATH)),
        os.path.exists(os.path.join(checkpoint_path, TRAINING_STATE_PATH)),
        os.path.exists(os.path.join(checkpoint_path, TRAINING_CONFIG_PATH))
    ])
    
def has_valid_checkpoint(output_dir: str) -> bool:
    """output_dir (str) : path to the training output directory which potentially contain multiple checkpointr"""
    list_chk = [p for p in os.listdir(output_dir) if p.startswith(CHECKPOINT_PREFIX)]
    if not list_chk:
        return False
    for chk_path in list_chk:
        chk_path = os.path.join(output_dir, chk_path)
        if not os.path.isdir(chk_path):
            continue
        if validate_checkpoint(chk_path):
            return True # only one valid checkpoint is enough

def get_latest_checkpoint(output_dir: str) -> str:
    """
    If no valid checkpoint is found, return None. Assume that the checkpoint directory is named as checkpoint_{%Y-%m-%d_%H-%M-%S}.
   
    Args: 
        output_dir (str) : path to the training output dir. 
    """
    if not has_valid_checkpoint(output_dir):
        return None
    
    sorted_paths =  sorted(
        [p for p in os.listdir(output_dir) if p.startswith(CHECKPOINT_PREFIX)],
        key=lambda x: os.path.getmtime(os.path.join(output_dir, x))
    )
    
    return os.path.join(output_dir, sorted_paths[-1])

def get_best_checkpoint(output_dir: str) -> str:
    """
    Similar to `get_latest_checkpoint`, but return the checkpoint with the best evaluation score. 
    """
    if not has_valid_checkpoint(output_dir):
        return None
    
    all_checkpoints = [p for p in os.listdir(output_dir) if p.startswith(CHECKPOINT_PREFIX)]
    # get training state for each checkpoint
    checkpoint_states = []
    for chk in all_checkpoints:
        with open(os.path.join(output_dir, chk, TRAINING_STATE_PATH), 'r') as f:
            state = pickle.load(f)
            score = state.best_score
            checkpoint_states.append((chk, score))
    # get the best checkpoint based on the score
    best_checkpoint = max(checkpoint_states, key=lambda x: x[1])
    return os.path.join(output_dir, best_checkpoint[0])

def rotating_checkpoint(output_dir: str, trial_name: str, max_keep: int = 8, keep_best: bool = False):
    """Rotate checkpoints by removing the oldest or worst checkpoint if the number exceeds `max_keep`."""
    output_dir = os.path.join(output_dir, trial_name)
    all_checkpoints = [p for p in os.listdir(output_dir) if CHECKPOINT_PREFIX in p]
    if len(all_checkpoints) <= max_keep or max_keep < 0:
        return

    checkpoint_states = []
    for chk in all_checkpoints:
        try:
            with open(os.path.join(output_dir, chk, TRAINING_STATE_PATH), 'rb') as f:
                state = pickle.load(f)
                checkpoint_states.append((chk, state.best_score))
        #file folder `checkpoint...` may have been created but not yet written the training state, etc. 
        except Exception as e:
            continue
    while len(all_checkpoints) > max_keep:
        if keep_best and checkpoint_states:
            worst_checkpoint = min(checkpoint_states, key=lambda x: x[1])
            shutil.rmtree(os.path.join(output_dir, worst_checkpoint[0]))
            checkpoint_states.remove(worst_checkpoint)
            all_checkpoints.remove(worst_checkpoint[0])
        else:
            oldest_checkpoint = min(all_checkpoints, key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))
            shutil.rmtree(os.path.join(output_dir, oldest_checkpoint))
            all_checkpoints.remove(oldest_checkpoint)
            checkpoint_states = [chk for chk in checkpoint_states if chk[0] != oldest_checkpoint]
    return

def save_checkpoint(save_on_each_node=False, model_safe_tensor=False, push_model_to_hub=False, **kwargs):
    """
    Save the training checkpoint including model state, optimizer state, scheduler state, random states, 
    and other training metadata. Optionally push the model to the hub. Expected kwargs are:
        - model : torch.nn.Module
        - optimizer : torch.optim.Optimizer
        - scheduler : torch.optim.lr_scheduler
        - training_state : TrainingState
        - training_config : TrainingConfig
        - data_loader (Optional): torch.utils.data.DataLoader or Tuple[torch.utils.data.DataLoader]
    """
    train_config = kwargs['training_config']
    train_config.token = None # we need to remove the token before saving the checkpoint as hugginface does not allow saving token
    train_config.api = None 
    chk_name =f"{CHECKPOINT_PREFIX}{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    chk_path = os.path.join(train_config.output_dir, train_config.trial_name, chk_name)
    os.makedirs(chk_path, exist_ok=True)
    # will either save the model state_dict or the model itself
    model_obj  = kwargs["model"] if model_safe_tensor else kwargs["model"].state_dict()
    model_path = MODEL_SAFE_TENSOR_PATH if model_safe_tensor else MODEL_PATH
    save(model_obj, os.path.join(chk_path, model_path), save_on_each_node=save_on_each_node)
    # Note that the `save` method will save a wrapped optimizer and scheduler: `AccelerateOptimizer` and `AccelerateScheduler`, so will use no serialization
    if "optimizer" in kwargs: 
        save(kwargs["optimizer"].state_dict(), os.path.join(chk_path, OPTIMIZER_PATH), save_on_each_node=save_on_each_node, safe_serialization=False)
    if "scheduler" in kwargs:
        save(kwargs["scheduler"].state_dict(), os.path.join(chk_path, SCHEDULE_PATH), save_on_each_node=save_on_each_node, safe_serialization=False)

    if "data_loader" in kwargs:
        if not isinstance(kwargs["data_loader"], Iterable): # incase we have train and eval dataloader
            data_loader = (kwargs["data_loader"], ) 
        for i, dl in enumerate(data_loader):
            if isinstance(dl.dataset, IterableDatasetShard):
                sampler = dl.get_sampler()
                if isinstance(sampler, SeedableRandomSampler):
                    sampler.save(os.path.join(chk_path, f"{DATALOADER_PATH}_{i}.bin"), save_on_each_node=save_on_each_node)
    
    random_state = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    random_state["torch_cuda_manual_seed"] = torch.cuda.get_rng_state_all()

    save(random_state, os.path.join(chk_path, "random_state.bin"), save_on_each_node=save_on_each_node)
    pickle.dump(kwargs["training_state"], open(os.path.join(chk_path, TRAINING_STATE_PATH), 'wb'))
    pickle.dump(kwargs["training_config"], open(os.path.join(chk_path, TRAINING_CONFIG_PATH), 'wb'))
    
    if hasattr(kwargs["model"], "config"): ####
        save(kwargs["model"].config, os.path.join(chk_path, "config.json"), save_on_each_node=save_on_each_node)
        
    rotating_checkpoint(train_config.output_dir, trial_name=train_config.trial_name, 
                        max_keep=train_config.max_checkpoint, keep_best=train_config.keep_best)
    try:
        save_checkpoint_to_repos_id(
            checkpoint_path=chk_path, config=train_config, api=kwargs.get("api", None)
        )
    except Exception as e:
        log_on_main(f"Error saving checkpoint to repos: {e}, ignoring...")
        
    return chk_path

def load_checkpoint(from_safe_tensor=False, **kwargs):
    """
    Load the training checkpoint including model state, optimizer state, scheduler state, random states,
    and other training metadata. Expected kwargs are:
        - checkpoint_path : str
        - model : torch.nn.Module
        - optimizer : torch.optim.Optimizer
        - scheduler : torch.optim.lr_scheduler
        - training_state : TrainingState
        - training_config : TrainingConfig
        - data_loader (Optional): torch.utils.data.DataLoader or Tuple[torch.utils.data.DataLoader]
    
    """
    chk_path = kwargs['checkpoint_path']

    map_location = kwargs.get('map_location', "cpu")
    if map_location is None: 
        if torch.cuda.is_available() and torch.distributed.get_world_size() > 1:    
            map_location = "on_device"
        else:
            map_location = "cpu" 
            
    if not from_safe_tensor: # will load state_dict
        kwargs['model'].load_state_dict(
            torch.load(os.path.join(chk_path, MODEL_PATH), map_location=map_location)
        )
    else: # this will load the model from safetensors
        kwargs['model'] = torch.load(os.path.join(chk_path, MODEL_SAFE_TENSOR_PATH), map_location=map_location)
        
    kwargs['optimizer'].load_state_dict(
        torch.load(os.path.join(chk_path, OPTIMIZER_PATH), map_location=map_location)
    )
    kwargs['scheduler'].load_state_dict(
        torch.load(os.path.join(chk_path, SCHEDULE_PATH), map_location=map_location)
    )
    train_state = pickle.load(open(os.path.join(chk_path, TRAINING_STATE_PATH), 'rb'))
    train_config = pickle.load(open(os.path.join(chk_path, TRAINING_CONFIG_PATH), 'rb'))
    
    for key, value in train_config.__dict__.items():
        setattr(kwargs['training_config'], key, value)
    for key, value in train_state.__dict__.items():
        setattr(kwargs['training_state'], key, value)

    if "data_loader" in kwargs:
        if not isinstance(kwargs["data_loader"], Iterable):
            data_loader = (kwargs["data_loader"], )
        
        for i, dl in enumerate(data_loader):
            if isinstance(dl.dataset, IterableDatasetShard):
                sampler = dl.get_sampler()
                if isinstance(sampler, SeedableRandomSampler):
                    loaded_sampler = torch.load(os.path.join(chk_path, f"{DATALOADER_PATH}_{i}.bin"))
                    sampler = dl.set_sampler(loaded_sampler)
    try:
        random_state = torch.load(os.path.join(chk_path, "random_state.bin"))
        random.setstate(random_state["random"])
        np.random.set_state(random_state["numpy"])
        torch.set_rng_state(random_state["torch"])
        if is_xpu_available():
            torch.xpu.set_rng_state_all(random_state["torch_xpu_manual_seed"])
        else:
            torch.cuda.set_rng_state_all(random_state["torch_cuda_manual_seed"])
        if is_torch_xla_available():
            xm.set_rng_state(random_state["xm_seed"])
            
    except Exception as e:
        pass # ignore random state loading error

def report_cuda_state():
    if not cuda.is_available() or not cuda.is_initialized():
        return 
    devices = cuda.device("cuda")
    return {
        "memory" : cuda.memory_usage(devices),
        "utilization" : cuda.utilization(devices),
        "temperature" : cuda.temperature(devices),
        "power_draw" : cuda.power_draw(devices),
        "clock_rate" : cuda.clock_rate(devices)  
    }

def report_speed_metrics(start_time, split="default", num_samples=None, num_steps=None, num_tokens=None): # hgf
    runtime = time.time() - start_time
    result = {f"{split}_runtime": round(runtime, 4)}
    
    if num_samples is not None:
        samples_per_second = num_samples / runtime
        result[f"{split}_samples_per_second"] = round(samples_per_second, 5)
    if num_steps is not None:
        steps_per_second = num_steps / runtime
        result[f"{split}_steps_per_second"] = round(steps_per_second, 5)
    if num_tokens is not None:
        tokens_per_second = num_tokens / runtime
        result[f"{split}_tokens_per_second"] = round(tokens_per_second, 5)
        
    return result

def push_model(ctx):
    '''Push Model to Hub'''
    if ctx._token is None or D.get_rank() != 0:
        return 
    try:
        ctx.model.module.save_pretrained(
            save_directory=ctx.output_trial_dir,
            push_to_hub=True,
            token=ctx._token,
        )
    except Exception as e:
        log_on_main(f"Error pushing model to Hub: {e}")

def validate_repo_exist(repo_id, repo_type=None, token=None):
    try:
        repo_info(repo_id, repo_type=repo_type, token=token)
        return True
    except RepositoryNotFoundError:
        return False
    
def save_checkpoint_to_repos_id(
    checkpoint_path: str, config: TrainingConfig, api: HfApi=None
): 
    if api is None:
        return
    repo_id = config.repo_id
    if not validate_repo_exist(config.repo_id, token=api.token):
        repo_id = api.create_repo(
            repo_id=config.repo_id,
            exist_ok=True).repo_id
    # get the `checkpoint_{%Y-%m-%d_%H-%M-%S}` part
    path_in_repo = re.search(r'checkpoint_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', checkpoint_path).group(0)
    
    api.upload_folder(
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        folder_path=checkpoint_path,
        commit_message=f'Add checkpointing: {checkpoint_path}',
    )

def load_checkpoint_repos(config: TrainingConfig):
    pass 