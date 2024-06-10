# TrainSLM
 Training Small Language Model
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


example: 

![alt text](https://github.com/pethai2004/TrainSLM/blob/main/tenb.png?raw=true)
![alt text](https://github.com/pethai2004/TrainSLM/blob/main/prof.png?raw=true)
