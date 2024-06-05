
from trainer import Trainer 

class Trainer(Trainer):
    
    def find_executable_batch_size(self, incremental=True, maximum_memory: float=0.95, *args, **kwargs):
        """Will change the batch size dynamically at the very begining of training. If `incremental` is True, it will start from batch size of 1, 
        else it will start from `self.config.per_device_batch_size`, and dynamically adjust the batch size. Call this instead of `Trainer.train()`.
        `maximum_memory` is the maximum memory to allocated"""

        _error = ["CUDA out of memory.",  # CUDA OOM
                  "DefaultCPUAllocator: can't allocate memory"] # CPU OOM
        device_max_memory = torch.cuda.get_device_properties(self.config.device).total_memory / 1e9
        should_allocate = int(device_max_memory * maximum_memory) 
        
        if not isinstance(self.train_dataset, ExecutableDataLoader):
            self.train_datast = ExecutableDataLoader()
        try: 
            self.train(*args, **kwargs)
        except Exception as e:
            pass 
 