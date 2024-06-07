# abstraction
import os 
import sys
import logging 
import time 
import torch 
from torch.nn.parallel import DistributedDataParallel
from torch import cuda
import torch.distributed as D
from torch.multiprocessing import spawn
from datasets import load_dataset, Dataset
from transformers import (
        PreTrainedTokenizerFast, 
        FalconConfig, 
        FalconForQuestionAnswering, 
        FalconForCausalLM,
    )

from src.trainer import Trainer
from src.config import TrainingConfig, TrainingState
from src.dist import init_worker, gather

logger = logging.getLogger(__name__)

############################################################################################################
def create_dummy_dataset_v0(tokenizer, length=728, max_preload=None):
    name_repo_id = "Owaner/UN_sessions"
    dataset = load_dataset(name_repo_id) 
    if max_preload is not None:
        dataset = dataset["train"].select(range(max_preload))
    else:
        dataset = dataset["train"]

    dataset = dataset.map(
        lambda x: tokenizer(
                x["text"],
        ),
        batch_size=1024,
        batched=True,
        num_proc=16,
        load_from_cache_file=True,
    )
    
    # this is not expensive operation so not need for parallelization
    rows = []
    current = []
    for each in dataset["input_ids"]:
        for token in each:
            current.append(token)
            if len(current) == length:
                rows.append(current)
                current = []
    dataset = Dataset.from_dict({"input_ids": rows}) # ignore other columns
    
    return dataset

def create_dummy_model_v0(vocab_size=6000):
    """Q and A model."""
    model_config = FalconConfig(
            vocab_size=vocab_size,
            num_hidden_layers=8,
            num_attention_heads=8,
            hidden_size=256,
            bias=True,
            attn_implementation="flash_attention_2",
        )
    return FalconForCausalLM(model_config)

############################################################################################################

class CausalTrainer(Trainer):
    
    def make_split_dataset(self, dataset):
        self.train_dataset, self.eval_dataset = self.split_train_eval(dataset)
    
    def sample(self, inputs):
        return None 
    
def main(dataset):
    
    if not D.is_initialized():
        D.init_process_group(backend="nccl")
    rank = D.get_rank()
    world_size = D.get_world_size()
    
    start_time_init = time.time()
    done_time = time.time() - start_time_init
    D.barrier()
    start_time_proc = time.time()
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="DefaultTokenizer")
    model = create_dummy_model_v0(len(tokenizer))
    
    training_config = TrainingConfig(
        trial_name="SecondTrial",
        num_epochs=5,
        per_device_batch_size=200,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        learning_rate=0.00001,
        device="cuda",
        max_training_examples=-1,
        max_training_steps=-1,
        token="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",
        num_processes=world_size,
        log_interval=3,
        checkpoint_interval=500,
        non_blocking=True,
        attn_implementation="flash_attention_2",
        mixed_precision="fp16",
        dispatch_on_device=False,
    )
    training_config._post_init()
    
    trainer = CausalTrainer(
        training_config=training_config,
        model=model,
        tokenizer=tokenizer
    )
    
    train_data, eval_data = trainer.split_train_eval(dataset)
    trainer.create_training_dataset(train_data)
    trainer.create_eval_dataset(eval_data)
    
    trainer.train(resume_from_checkpoint=False)
    
    D.destroy_process_group()
    
    
if __name__ == "__main__":

    dataset = create_dummy_dataset_v0(
        tokenizer=PreTrainedTokenizerFast(tokenizer_file="DefaultTokenizer"),
        max_preload=None
    )
    main(dataset)
    #torchrun --nproc_per_node=4

