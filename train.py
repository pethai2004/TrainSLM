# abstraction
import os 
import sys
import logging 
import time 
import transformers as tfm 
import torch.distributed as D

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

from src.trainer import Trainer
from src.config import TrainingConfig
from common import create_dummy_dataset_v0, create_dummy_model_v0

logger = logging.getLogger(__name__)
class CausalTrainer(Trainer):

    def sample(self, inputs):
        inputs["labels"] = inputs["input_ids"]
        inputs["use_cache"] = False

        return super().sample(inputs)
    
def main(dataset):
    
    if not D.is_initialized():
        D.init_process_group(backend="nccl")
    rank = D.get_rank()
    world_size = D.get_world_size()
    
    start_time_init = time.time()
    done_time = time.time() - start_time_init
    D.barrier()
    start_time_proc = time.time()
    tokenizer = tfm.PreTrainedTokenizerFast(tokenizer_file="DefaultTokenizer")
    model = create_dummy_model_v0(len(tokenizer))
    
    training_config = TrainingConfig(
        trial_name="ThirdTrial",
        num_epochs=8,
        per_device_batch_size=150,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        learning_rate=0.0001,
        device="cuda",
        max_training_examples=-1,
        max_training_steps=-1,
        token="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",
        num_processes=world_size,
        log_interval=20,
        checkpoint_interval=500,
        non_blocking=True,
        attn_implementation="flash_attention_2",
        # pin_memory=True,
        mixed_precision="fp16",
        dispatch_on_device=True,
    )
    training_config._post_init()
    
    trainer = CausalTrainer(
        training_config=training_config,
        model=model,
        tokenizer=tokenizer
    )
    trainer._should_log_flops = True
    train_data, eval_data = trainer.split_train_eval(dataset)
    trainer.create_training_dataset(train_data)
    trainer.create_eval_dataset(eval_data)
    
    #trainer.run_on_profiler()
    trainer.train(resume_from_checkpoint=False)
    D.destroy_process_group()
    
    
if __name__ == "__main__":
    
    dataset = create_dummy_dataset_v0(
        tokenizer=tfm.PreTrainedTokenizerFast(tokenizer_file="DefaultTokenizer"),
        max_preload=None
    )
    main(dataset)

