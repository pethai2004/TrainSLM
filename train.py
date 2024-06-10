
import os 
import logging 
import transformers as tfm 
import torch.distributed as D

from src.config import TrainingConfig, parse_arg
from src.utility import get_tokenizer
from common import create_fineweb_dataset, load_one_fineweb_dataset, CausalTrainer, create_fineweb_model

logger = logging.getLogger(__name__)
    
def main(dataset, 
         tokenizer: tfm.PreTrainedTokenizerFast, 
         model: tfm.PreTrainedModel, 
         training_config: TrainingConfig):
    
    if not D.is_initialized():
        D.init_process_group(backend="nccl")
    D.barrier()
    
    trainer = CausalTrainer(
        training_config=training_config,
        model=model,
        tokenizer=tokenizer
    )
    trainer._should_log_flops = False # still error
    #[rank3]:     return input_dict[self.main_input_name].numel()
#[rank3]: AttributeError: 'list' object has no attribute 'numel'
    train_data, _ = trainer.split_train_eval(dataset)
    trainer.create_training_dataset(train_data)
    
    trainer.train(resume_from_checkpoint=True)
    D.destroy_process_group()
   
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", type=str, default="fineweb")
    parser.add_argument("--cache_dir", type=str, default="fineweb_cache")
    parser.add_argument("--max_preload", type=int, default=-1)
    parser.add_argument("--seq_length", type=int, default=512)
    
    args, remaining_argv = parser.parse_known_args()

    # parse TrainingConfig
    config_parser = argparse.ArgumentParser()
    training_config = parse_arg(config_parser, args=remaining_argv)
    
    tok = "/root/TrainSLM/DefaultTokenizer"
    abs_path = os.path.abspath(tok)
    tokenizer = get_tokenizer(tok=tok)
    
    assert tokenizer.pad_token_id is not None
    assert tokenizer.eos_token_id is not None
    
    if not os.path.exists(args.local_dir): # have not loaded the dataset yet 
        print(f"Loding dataset to {args.local_dir}")
        load_one_fineweb_dataset(
            local_dir=args.local_dir,
            subset="CC-MAIN-2024-10",
            parquet_file="train-00000-of-00020.parquet",
        )
    dataset = create_fineweb_dataset(
        tokenizer=tokenizer,
        local_dir=args.local_dir,
        cache_dir=args.cache_dir,
        seq_length=args.seq_length,
        max_preload=args.max_preload
    )
    print(f"Done creating dataset with {len(dataset)} samples")
    
    model = create_fineweb_model()
    training_config.token = "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"

    training_config.dispatch_on_device = True
    training_config.num_epochs = 20
    training_config.num_profile_steps = 10
    training_config.log_interval = 20
    training_config.checkpoint_interval = 500
    training_config.per_device_batch_size = 150
    
    training_config.max_checkpoint = 3
    
    main(
        dataset=dataset,
        tokenizer=tokenizer,
        model=model,
        training_config=training_config
    )
