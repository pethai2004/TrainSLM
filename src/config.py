# This file contains the configuration # TODO: add ZeroRedundancyOptimizer
import os 
import psutil
import argparse
import math
import json
import logging
import torch
from typing import Union, Dict, Optional
from dataclasses import dataclass, field, asdict, fields

import torch
import torch.distributed as D

logger = logging.getLogger(__name__)
def log_on_main(*args, **kwargs):
    if D.get_rank() == 0:
        logging.getLogger().warning(*args, **kwargs)
        
CHECKPOINT_PREFIX = 'checkpoint_'
MODEL_SAFE_TENSOR_PATH = 'model.safetensors'
MODEL_PATH = 'model.bin'
OPTIMIZER_PATH = 'optimizer.bin'
SCHEDULE_PATH = 'schedule.bin'
TRAINING_STATE_PATH = 'training_state'
TRAINING_CONFIG_PATH = 'training_config'
DATALOADER_PATH = 'dataloader.bin'
tensorboard_log_dir = 'logs'
profiler_log_dir = 'profiler'
_default_port = 6006 # default port for tensorboard
non_decay_cls = (torch.nn.LayerNorm, torch.nn.Embedding)
default_optim_cls = torch.optim.AdamW
default_scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
default_lr_kwargs = {"eta_min": 0.000002, "num_cosine_restart_factor": 200,"T_mult": 2}
ignore_padding = -100

def _set_global_config_var(name, new_value): 
    # define all the global variables here
    global CHECKPOINT_PREFIX, MODEL_SAFE_TENSOR_PATH, MODEL_PATH, OPTIMIZER_PATH, SCHEDULE_PATH, TRAINING_STATE_PATH, TRAINING_CONFIG_PATH, DATALOADER_PATH, tensorboard_log_dir, _default_port, non_decay_cls, default_optim_cls, default_scheduler_cls, default_lr_kwargs, ignore_padding
    if name in globals():
        globals()[name] = new_value
    else: log_on_main(f"Global variable {name} not found, ignoring.")

config_instance_ = None
state_instance_ = None

def get_config() -> "TrainingConfig":
    """Returns the singleton instance of TrainingConfig."""
    global config_instance_
    if config_instance_ is None:
        raise RuntimeError("TrainingConfig instance not initialized.")
    return config_instance_

@dataclass
class TrainingState:
    
    trial_name : str = field(default="DefaultTrialName", metadata={"help": "Name of the training trial."})
    global_training_step : int = field(default=0, metadata={"help": "Global training step, counted in gradient updates."})
    eval_step : int = field(default=0, metadata={"help": "Evaluation step, counted as number of batches."})
    global_epoch : int = field(default=0, metadata={"help": "Global epoch."})
    best_score : float = 0.0
    is_in_train : bool = False 
    num_batch_training_so_far : int = field(default=0, metadata={"help": "Number of batches trained so far."})
    num_tokens_so_far : int = 0
    num_steps_per_epoch : int = field(default=-1, metadata={"help": "Number of gradient updates per epoch."})
    loss : float = 0.0

    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is not None:
            log_on_main("TrainingState instance already exists.")
        cls._instance = super(TrainingConfig, cls).__new__(cls)
        global state_instance_
        state_instance_ = cls._instance
        return cls._instance

@dataclass
class TrainingConfig: #TODO: should `num_processes` be set here since we cannot control from here?
    '''
    Available Attributes and property after post init:
        - `optimizer_cls`: Optimizer class (default: torch.optim.AdamW)
        - `scheduler_cls`: Scheduler class (default: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)

    main property:
        - `precision`: Precision of the model (torch.float16, torch.float32)
        - `batch_size`: Batch size per process
        - `full_batch_size`: Full batch size (per_device_batch_size * num_processes * gradient_accumulation_steps)

    '''
    output_dir: str = field(default="DefaultOutputDir", metadata={"help": "Directory for saving outputs."})
    trial_name: str = field(default="DefaultTrialName", metadata={"help": "Name of the training trial."})
    model_name_or_path: str = field(default="", metadata={"help": "Path to local directory or name of huggingface repo_id of the model."})  
    model_cls_name: str = field(default="", metadata={"help": "Class name of the model."})
    optimizer_cls_name: str = field(default="", metadata={"help": "Class name of the optimizer."})
    scheduler_cls_name: str = field(default="", metadata={"help": "Class name of the scheduler."})
    model_config: Dict = field(default=dict, metadata={"help": "Configuration dictionary for the model."})
    optimizer_kwargs: Dict = field(default=dict, metadata={"help": "Keyword arguments for the optimizer."})
    scheduler_kwargs: Dict = field(default=dict, metadata={"help": "Keyword arguments for the scheduler."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay applied to Optimizer."})
    per_device_batch_size: int = field(default=64, metadata={"help": "Batch size per processes: full_batch_size = per_device_batch_size * num_gpu."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Steps for gradient accumulation."})
    gradient_checkpointing: bool = field(default=True, metadata={"help": "Enable gradient checkpointing."})
    
    num_epochs: int = field(default=3, metadata={"help": "Number of training epochs."})
    max_training_steps: int = field(default=-1, metadata={"help": "Maximum training steps per epoch, measured in gradient updates step."})
    max_training_examples: int = field(default=-1, metadata={"help": "Maximum number of training examples. If -1, use all examples."})
    max_eval_steps: int = field(default=-1, metadata={"help": "Maximum evaluation steps. (is number of batches), if -1, use all examples. If 0, do not evaluate."})
    gradient_clip_norm: Union[int, str] = field(default=3.0, metadata={"help": "Norm for gradient clipping, if -1 or `inf`, no clipping."})
    gradient_clip_value: Union[int, str] = field(default=3.0, metadata={"help": "Value for gradient clipping, if -1, `inf` clipping."})
    tokenizer_name_or_path: str = field(default="", metadata={"help": "Path to local directory or name of huggingface repo_id of the tokenizer."})
    mixed_precision: str = field(default="fp16", metadata={"help": "Whether to use mixed precision training. Available options: 'fp16', 'bfp16', 'off'."})
    device: str = field(default="cuda", metadata={"help": "Device for training. Note that this script is only supported on GPU. (so do not change this)"})
    seed: int = field(default=42, metadata={"help": "Seed for random number generation."})
    seed_for_each_worker : bool = field(default=False, metadata={"help": "Each worker will have a different seed: seed_worker = config.seed + rank."}) 
    full_determinism: bool = field(default=False, metadata={"help": "Ensure full determinism in training. Discouraged since it may slow down training."})
    learning_rate: float = field(default=5e-5, metadata={"help": "Initial learning rate."})
    minimum_learning_rate: float = field(default=1e-5, metadata={"help": "Minimum learning rate."})
    lr_update_strategy: str = field(default="step", metadata={"help": "Strategy for updating learning rate. Available options: 'step', 'epoch', 'plateau'. (currently not supported)"})
    repo_id: str = field(default="", metadata={"help": "Repo ID for pushing to hub checkpointing. If empty string is provided, do not push."})
    token: str = field(default="", metadata={"help": "Token for hub access. Must be provided if repo_id is provided."})
    disable_hf_progress_bar: bool = field(default=True, metadata={"help": "Disable Hugging Face progress bar."})
    model_max_shard_size: int = field(default=1, metadata={"help": "Maximum model shard size (in GB)."})
    max_checkpoint: int = field(default=8, metadata={"help": "Maximum number of checkpoints. If -1, keep all checkpoints. If 0, do not save checkpoints."})
    keep_best: bool = field(default=False, metadata={"help": "Keep only the best checkpoints when the maximum number of checkpoints is reached, otherwise will keep the latest checkpoints."})
    checkpoint_interval: int = field(default=1000, metadata={"help": "Interval for saving checkpoints. If 0, do not save checkpoints."})
    push_to_hub_interval: int = field(default=1000, metadata={"help": "Interval for pushing to hub. If -1, will set to checkpoint_interval. If 0, do not push to hub."})
    log_interval: int = field(default=50, metadata={"help": "Interval for logging to console. If 0, do not log. Default to 50"})
    tensorboard_interval: int = field(default=1, metadata={"help": "Interval for logging to TensorBoard. If 0, do not log to TensorBoard. If 1, log every step. If -1, log according to log_interval. Default to 1"})
    
    past_input_name: str = field(default="", metadata={"help": "Name of the past input."})
    model_input_name: list = field(default=("input_ids",), metadata={"help": "A list of model input names."})
    
    num_processes: int = field(default=-1, metadata={"help": "Total number of GPUs. If None or -1, will use all available GPUs."})
    neftune_noise_alpha: float = field(default=0.0, metadata={"help": "Noise alpha for Neftune."})
    save_safe_tensor: bool = field(default=False, metadata={"help": "Save tensor in a safe format."})
    train_test_split: Union[float, int] = field(default=0.05, metadata={"help": "Split ratio for training and testing."})
    num_dataloader_workers: int = field(default=4, metadata={"help": "Number of data loader workers."})
    log_on_each_node: bool = field(default=False, metadata={"help": "Log on each node."})
    report_nan: bool = field(default=True, metadata={"help": "Report NaNs or zero of the gradients, loss, model output, or model parameters."})
    attn_implementation: str = field(default="sdpa", metadata={"help": "Attention implementation type. Available options: 'sdpa', 'flash_attention_2', 'eager'."})
    pin_memory: bool = field(default=True, metadata={"help": "Pin memory for DataLoader."})
    non_blocking: bool = field(default=True, metadata={"help": "Non-blocking data transfer argument to `to` method."})
    dispatch_on_device : bool = field(default=False, metadata={"help": "Automatically dispatch the input to appropriate device`"})
    persistent_workers : bool = field(default=False, metadata={"help": "Keep the workers alive between iterations. Argument to `DataLoader`."})
    enable_hp_search: bool = field(default=False, metadata={"help": "Enable hyperparameter search for predefined default hyperparameters. (currently not supported)"})
    num_profile_steps: int = field(default=5, metadata={"help": "Number of steps for profiling. If 0, do not profile."})
    
    is_post_init = False 
    
    def _post_init(self, force=False):
        """Post initialization and sanity checks."""
        if self.is_post_init and not force:
            return
        
        for key, value in asdict(self).items():
            if value == "":
                setattr(self, key, None)# will set the value "" to None (need better way of parsing None)
            if value is dict:
                setattr(self, key, {})
                
        if self.disable_hf_progress_bar: 
            os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
            os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'
            
        if self.token is None and self.repo_id is not None:
            token = os.getenv('HF_TOKEN', None)
            if token is None:
                raise ValueError("Either provide a token or set the environment variable HF_TOKEN \n \
                                    if.repo_id is set.")
            self.token = token
        
        assert self.device in ["cuda", "mps", "cpu"], "Only cuda, mps, and cpu devices are supported."
        
        if not D.is_initialized():
            if torch.cuda.is_available() and self.device == "cuda":
                if self.num_processes == -1:
                    self.num_processes = torch.cuda.device_count() 
                self.num_processes = min(self.num_processes, torch.cuda.device_count())
            else: 
                if self.device == "cuda":
                    log_on_main("CUDA is not available. Fallback to MPS.")
                    self.device = "mps"
                
                if self.num_processes == -1:
                    self.num_processes = psutil.cpu_count(logical=True)
                self.num_processes = min(self.num_processes, psutil.cpu_count(logical=True))
                # thus num-thread will be torch.set_num_threads(floor(num_thread / num_processess))
        else: 
            self.num_processes = D.get_world_size()
            # will ignore setting device if D is initialized
        log_on_main(f"Number of processes: {self.num_processes}")

        self.per_device_batch_size = max(1, self.per_device_batch_size)
        if self.per_device_batch_size % self.num_processes != 0: 
            while self.per_device_batch_size % self.num_processes != 0:
                self.per_device_batch_size -= 1 # reduce until divisible by num_processes
            log_on_main(f"Provided per_device_batch_size is not divisible by num_processes. Adjusted to {self.per_device_batch_size}")
            
        if self.model_cls_name is not None and self.model_name_or_path is not None:
            self.model_cls_name = None
            log_on_main("Both model_cls_name and model_name_or_path are provided. set model_cls_name to None.")
        if self.model_cls_name is not None and(self.model_config is None or self.model_config == {}):
            raise ValueError("model_config must be provided if model_cls_name is provided.")
        # lr
        if self.optimizer_kwargs.get("lr", None) is not None:
            if self.learning_rate != self.optimizer_kwargs["lr"]:
                log_on_main(f"Found incongruence between `learning_rate` and `optimizer_kwargs['lr']`. Setting `learning_rate` to `optimizer_kwargs['lr']`.")
                self.learning_rate = self.optimizer_kwargs["lr"]
        else: 
            self.optimizer_kwargs["lr"] = self.learning_rate
            log_on_main(f"`optimizer_kwargs['lr']` is not provided. Setting `optimizer_kwargs['lr']` to `learning_rate`.")
        # weight decay (same logic as lr)
        if self.optimizer_kwargs.get("weight_decay", None) is not None:
            if self.weight_decay != self.optimizer_kwargs["weight_decay"]:
                log_on_main(f"Found incongruence between `weight_decay` and `optimizer_kwargs['weight_decay']`. Setting `weight_decay` to `optimizer_kwargs['weight_decay`.")
                self.weight_decay = self.optimizer_kwargs["weight_decay"]
        else:
            self.optimizer_kwargs["weight_decay"] = self.weight_decay
            log_on_main(f"`optimizer_kwargs['weight_decay']` is not provided. Setting `optimizer_kwargs['weight_decay']` to `weight_decay`.")
        
        if self.optimizer_cls_name is not None:
            try:
                self.optimizer_cls = getattr(torch.optim, self.optimizer_cls_name)
            except AttributeError:
                raise ValueError(f"Optimizer class {self.optimizer_cls_name} not found in torch.optim.")
        else:
            self.optimizer_cls = default_optim_cls
            self.optimizer_cls_name = self.optimizer_cls.__name__
            self.optimizer_kwargs["amsgrad"] = True
            
            log_on_main(f"Optimizer class is not provided. Using default optimizer: {self.optimizer_cls_name}")
        
        if self.scheduler_cls_name is not None:
            try:
                self.scheduler_cls = getattr(torch.optim.lr_scheduler, self.scheduler_cls_name)
            except AttributeError:
                raise ValueError(f"Scheduler class {self.scheduler_cls_name} not found in torch.optim.lr_scheduler.")
        else:
            self.scheduler_cls = default_scheduler_cls
            self.scheduler_cls_name = self.scheduler_cls.__name__
            self.scheduler_kwargs["eta_min"] = self.scheduler_kwargs.get("eta_min", default_lr_kwargs["eta_min"])
            num_cosine_restart = max(int(self.max_training_steps // default_lr_kwargs["num_cosine_restart_factor"]), 1)
            self.scheduler_kwargs["T_0"] = self.scheduler_kwargs.get("T_0", num_cosine_restart)
            self.scheduler_kwargs["T_mult"] = self.scheduler_kwargs.get("T_mult", default_lr_kwargs["T_mult"])
            log_on_main(f"Scheduler class is not provided. Using default scheduler: {self.scheduler_cls_name}")
        
        if self.tokenizer_name_or_path is None:
            log_on_main("Tokenizer name or path is not provided") # just a warning

        self.log_interval = max(50, self.log_interval)
        
        if self.tensorboard_interval == -1:
            self.tensorboard_interval = self.log_interval
            log_on_main(f"Tensorboard interval is not provided. Automatically set to log_interval: {self.tensorboard_interval}")
            
        if self.push_to_hub_interval > 0 and self.repo_id is None:
            self.repo_id = f"{self.output_dir}_{self.trial_name}"
            log_on_main(f"Push to hub interval is provided but repo_id is not provided, default repo_id to `{self.repo_id}`")
        elif self.push_to_hub_interval == -1:
            self.push_to_hub_interval = self.checkpoint_interval
            log_on_main(f"Push to hub interval is not provided. Automatically set to checkpoint_interval: {self.push_to_hub_interval}")
            
        if self.gradient_clip_norm == -1 or self.gradient_clip_norm == "inf":
            self.gradient_clip_norm = 0.0
        if self.gradient_clip_value == -1 or self.gradient_clip_value == "inf":
            self.gradient_clip_value = 0.0
        
        if self.lr_update_strategy not in ["step", "epoch", "plateau"]:
            raise ValueError(f"Invalid lr_update_strategy: {self.lr_update_strategy}, available options: 'step', 'epoch', 'plateau'.")
        
        if not isinstance(self.model_input_name, (tuple, list)):
            if isinstance(self.model_input_name, str):
                self.model_input_name = [self.model_input_name]
            else:
                raise ValueError("model_input_name must be a list, tuple, or string. But got: {self.model_input_name}")
        
        if self.mixed_precision == "bfp16":
            if self.device == "cpu":
                raise ValueError("BFloat16 is not supported on CPU.")
            
        if self.attn_implementation not in ["sdpa", "flash_attention_2", "eager"]:
            raise ValueError(f"Invalid attention implementation: {self.attn_implementation}, available options: 'sdpa', 'flash_attention_2', 'eager'.")
        
        assert self.num_profile_steps >= 0, "num_profile_steps must be non-negative."
        log_on_main(f"Done config post init")
        self.is_post_init = True
        
    @property
    def precision(self):
        assert hasattr(self, "is_post_init"), "Call _post_init() first."
        if self.mixed_precision == "fp16":
            return torch.float16
        elif self.mixed_precision == 'bfp16':
            return torch.bfloat16 
        elif self.mixed_precision == "off":
            return torch.float32 # always training in fp32
        else:
            raise ValueError(f"Invalid mixed precision: {self.mixed_precision}, available options: 'fp16', 'bfp16', 'off'.")
        
    @property
    def print_option(self):
        if not hasattr(self, "is_post_init"):
            self._post_init()
        reprs = {
            "output_dir" : self.output_dir,
            "trial_name" : f"{self.trial_name}",
            "optimizer" : self.optimizer_cls_name,
            "scheduler" : self.scheduler_cls_name,
            "model" : self.model_cls_name,
            "initial_learning_rate" : self.learning_rate,
        }
        return reprs

    def pring_all_field(self):

        for f in fields(TrainingConfig):
            print(f"{f.name:<30}: {getattr(self, f.name)}")

    def save(self, output_dir=None):
        """Save the configuration to a JSON file."""
        if output_dir is None:
            output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(asdict(self), f, indent=4)

    @classmethod
    def load(cls, config_path):
        """Load the configuration from a JSON file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def batch_size(self):
        """Batch size: per_device_batch_size * num_processes."""
        return self.per_device_batch_size * max(1, self.num_processes)
    
    @property
    def full_batch_size(self):
        """Full batch size: per_device_batch_size * num_processes * gradient_accumulation_steps."""
        return self.per_device_batch_size * max(1, self.num_processes) * self.gradient_accumulation_steps
    
    @property
    def tensorboard_path(self):
        return os.path.join(self.output_dir, self.trial_name, tensorboard_log_dir)
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is not None:
            log_on_main("TrainingState instance already exists.")
        cls._instance = super(TrainingConfig, cls).__new__(cls)
        global config_instance_ 
        config_instance_ = cls._instance 
        return cls._instance
    
def parse_arg(parser, config=None, args=None):
    """Parse the arguments and set the fields of the TrainingConfig."""
    if config is None:
        config = TrainingConfig()

    for f in fields(config):
        help_text = f.metadata.get("help", "No help available")
        parser.add_argument(f'--{f.name}', type=type(f.default), default=f.default, help=help_text)

    args = parser.parse_args(args)
    parsed_dict = vars(args)

    for field in parsed_dict:
        # if key is the field of the TrainingConfig, set it, else ignore
        if field in asdict(config) and parsed_dict[field] is not None:
            print(f"Setting {field} to {parsed_dict[field]}")
            setattr(config, field, parsed_dict[field])

    return config