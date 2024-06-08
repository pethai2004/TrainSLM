
from transformers import (
        PreTrainedTokenizerFast, 
        FalconConfig, 
        FalconForQuestionAnswering, 
        FalconForCausalLM,
    )

from datasets import load_dataset, Dataset

def create_dummy_model_v0(vocab_size=6000):
    """Q and A model."""
    model_config = FalconConfig(
            vocab_size=vocab_size,
            num_hidden_layers=10,
            num_attention_heads=16,
            hidden_size=128,
            bias=True,
            attn_implementation="flash_attention_2",
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

def create_dummy_dataset_v1(tokenizer, length=1024, max_preload=None):
    pass 