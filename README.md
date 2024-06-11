# TrainSLM
 Training Small Language Model

You can view example model using this script on Hugginface Model: 
`https://huggingface.co/Owaner/fineweb-falcon?text=Once+upon+a+time%2C` 

Using Trainer:

   The `Trainer` class implement complete training script for Data Distributed Training scheme, and 
   it the future support Pipeline Parallelism. It support training on CUDA and MPS only. 

   To use `Trainer`, one must first passed the training_dataset and optional eval_dataset through:
       1. Constructor as kwargs
       2. Override the `create_training_dataset` or `create_eval_dataset` and call its super, it should return 
           an instance of `datasets.Dataset` or `datasets.IterableDataset`, otherwise not currently support.
   
   The `config.TrainingConfig` holds all necessary training configuration.


Example:

   I provided three full pretraining example including:
       - Code Generation
       - Q and A (using Pipeline Parallelism) with some modification on `Trainer`
       - Text to Speech

Tip Note: 

   > When providing custome dataset, it should be strictly of type `datasets.Dataset`.
   > when launching Tensorboard, it should be log to path `config.output_dir/config.trial_name`. This will
       allow to mornitor different trial single-handledly. 

Future Implementation:
    > Multiple Node Data Distributed scheme
    > Fine Tuning


Run : 

'chmod +x setup.sh'
source setup.sh

then 
- `torchrun --nproc-per-node=4 train.py --per_device_batch_size=128`

Load example Model Trained on FineWeb using Falcon model from scratch

```
import transformers as tfm 

model = tfm.AutoModelForCausalLM.from_pretrained("Owaner/fineweb-falcon")
tokenizer = tfm.PreTrainedTokenizerFast.from_pretrained("Owaner/falcon_tokenizer")

example = "When habitually indulge in "
tokenized_input = tokenizer(example, return_tensors="pt", return_token_type_ids=False)
output = model.generate(
    inputs=tokenized_input["input_ids"],
    attention_mask=tokenized_input["attention_mask"],
    do_sample = True,
    max_length=100,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    num_return_sequences=5
)
output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

for i, o in enumerate(output_text):
    print(f"Output {i+1}: {o}")
```

example: 

![alt text](https://github.com/pethai2004/TrainSLM/blob/main/tenb.png?raw=true)
![alt text](https://github.com/pethai2004/TrainSLM/blob/main/prof.png?raw=true)
