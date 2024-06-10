import os

from huggingface_hub import snapshot_download
from transformers import (
        PreTrainedTokenizerFast, 
        FalconConfig, 
        FalconForCausalLM,
    )

from datasets import load_dataset, Dataset
from preprocess_text import regex_preprocessor

from src.data_utility import constant_length_dataset_with_sort
from src.trainer import Trainer

def create_fineweb_model(vocab_size=6000, attn_implementation="flash_attention_2"):
    """Q and A model."""
    model_config = FalconConfig(
            vocab_size=vocab_size,
            num_hidden_layers=12,
            num_attention_heads=16,
            hidden_size=512,
            hidden_dropout=0.05,
            attention_dropout=0.05,
            parallel_attn=False,
            bias=True,
            attn_implementation=attn_implementation,
        )
    return FalconForCausalLM(model_config)

def create_dummy_dataset_v0(tokenizer, length=512, max_preload=None):
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

def load_one_fineweb_dataset(
        local_dir = "fineweb",
        subset="CC-MAIN-2024-10",
        parquet_file="train-00000-of-00020.parquet",
    ):
    '''Load one parquet file from the fineweb dataset (about 2GB with 0.5b token)'''
    name_repo_id = "HuggingFaceFW/fineweb-edu"
    
    folder = snapshot_download(
                repo_id=name_repo_id,
                repo_type="dataset",
                local_dir=f"./{local_dir}/",
                allow_patterns=f"data/{subset}/{parquet_file}"
    )
    return folder

def create_fineweb_dataset(
    tokenizer: PreTrainedTokenizerFast,
    local_dir="fineweb",
    subset="CC-MAIN-2024-10",
    cache_dir="fineweb_cache",
    apply_deconcatenate=True,
    seq_length=512,
    max_preload=-1
):
    path_to_parq = os.path.join(os.path.abspath(local_dir), "data", subset) # ./fineweb/data/CC-MAIN-2024-10/*.parquet
    parquet_list_paths = [
        os.path.join(path_to_parq, each) 
        for each in os.listdir(path_to_parq) 
    ]
    
    dataset = Dataset.from_parquet(
        parquet_list_paths,
        cache_dir=cache_dir,
    )
    
    if max_preload != -1 and len(dataset) > max_preload:
        print(f"Truncating dataset from {len(dataset)} to {max_preload}")
        dataset = dataset.select(range(max_preload))
        
        
    cols = dataset.column_names
    cols.remove("text")
    dataset = dataset.remove_columns(cols)
    dataset = dataset.map(
        lambda x: regex_preprocessor(x),
        batched=True,
        num_proc=16,
        load_from_cache_file=True,
        drop_last_batch=True,
        batch_size=2024,
    )
    dataset = dataset.map(
        lambda x: tokenizer(
                x["text"],
                padding="do_not_pad",
                truncation=False,
                return_length = True,
        ),
        batch_size=2024,
        batched=True,
        num_proc=16,
        load_from_cache_file=True,
    )
    
    dataset = dataset.remove_columns("text")
    dataset = dataset.remove_columns("length") # the `length` column is now irrelevant as we resize all of the samples
    dataset = dataset.remove_columns('token_type_ids')
    if apply_deconcatenate:
        dataset = dataset.map(
            lambda x: constant_length_dataset_with_sort(
                x=x,
                seq_length=seq_length,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            ),
            batched=True,
            num_proc=16,
            batch_size=2024,
        ) 
    # this is needed since when concat, we may leave out empty list
    dataset = dataset.filter(lambda x: len(x["input_ids"]) >= seq_length, num_proc=16)
    
    return dataset

class CausalTrainer(Trainer):
    
    def sample(self, inputs):
        # 1: not masked
        inputs["labels"] = inputs["input_ids"]
        inputs["use_cache"] = False

        return super().sample(inputs)

