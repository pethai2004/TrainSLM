
import os 
import logging 
import numpy as np
import random
import torch

import datasets
from datasets import Dataset as hgf_Dataset
import transformers as tfm
from transformers import PreTrainedTokenizerFast, AutoModel
from transformers.models.auto.modeling_auto import MODEL_MAPPING
from tokenizers import Tokenizer

from config import * 

logger = logging.getLogger(__name__)

def activate_neftune(ctx, model):
    if ctx.config.neftune_noise_alpha is None or ctx.config.neftune_noise_alpha == 0.0:
        return model
    embeddings = model.get_input_embeddings()
    embeddings.neftune_noise_alpha = ctx.config.neftune_noise_alpha
    ctx.neftune_hook_handle = embeddings.register_forward_hook(neftune_forward_hook)
    ctx._activated_neftune = True
    logging.getLogger().warning(f"Neftune noise alpha is activated with value: {ctx.config.neftune_noise_alpha}")
    return model

def deactivate_neftune(ctx, model):
    if not ctx._activated_neftune:
        return model
    embeddings = model.get_input_embeddings()

    ctx.neftune_hook_handle.remove()
    del embeddings.neftune_noise_alpha, unwrapped_model
    logging.getLogger().warning(f"Deactivate Neftune noise alpha.")
    
def neftune_forward_hook(module, input, output):

    if module.training:
        dims = torch.tensor(output.size(1) * output.size(2))
        mag_norm = module.neftune_noise_alpha / torch.sqrt(dims)
        output = output + torch.zeros_like(output).uniform_(-mag_norm, mag_norm)
    return output

def get_parameter_names(model):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"for n in get_parameter_names(child)
        ]
    result += list(model._parameters.keys())
    return result

def set_seed(seed: int, full_determinism: bool=False): # this function do not do tf-seed related stuff
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if full_determinism:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True)

        # Enable CUDNN deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_tokenizer(tok):
    """Get tokenizer from provided path or object."""
    special_tokens = {
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "eos_token": "[EOS]",
        "mask_token": "[MASK]"
    }
    
    tokenizer = None 
    if isinstance(tok, (Tokenizer, PreTrainedTokenizerFast)):
        if isinstance(tok, Tokenizer):
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok)
        tokenizer.add_special_tokens(special_tokens)
    elif isinstance(tok, str):
        try:
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(tok))
        except:
            try:
                tokenizer = PreTrainedTokenizerFast.from_pretrained(tok) 
            except:
                raise ValueError("Cannot get tokenizer")   
        tokenizer.add_special_tokens(special_tokens)

    return tokenizer

#TODO: assert list(tfm.models.auto.modeling_auto.MODEL_MAPPING.values())
def create_model(config, model=None, _init_method=None, **_init_args) -> torch.nn.Module:
    """Naive create model for Trainer._init_setup method."""
    _model = None 
    if model is not None:
        if isinstance(model, torch.nn.Module):
            _model = model
    elif config.model_cls_name is not None and config.model_config is not None:
        if hasattr(config.model_config, "_attn_implementation"):
            config.model_config._attn_implementation = config.attn_implementation
            logger.warning(f"Autoset `config.model_config._attn_implementation` to {config.attn_implementation}")
        try:
            _model = config.model_cls_name(config.model_config)
        # if raise ValueError or ImportError, change attn_implementation to None and try again
        except (ValueError, ImportError) as e:
            config.model_config._attn_implementation = None
            logger.warning(f"Error when trying to initialize model with `config.attn_implementation` = {config.attn_implementation}. \
                Try to initialize model without it")
            _model = config.model_cls_name(config.model_config)
        except Exception as e:
            raise e
        
    elif config.model_name_or_path is not None: 
        model = AutoModel.from_pretrained(config.model_name_or_path)
    
    elif _init_method is not None:
            model = _init_method(model, **_init_args)
    else: 
        raise ValueError("Model must be provided as:\n \
                                    1. In the constructor \n \
                                    2. Provided `config.model_name_or_path` \n \
                                    3. Provide both config.model_config and config.model_cls_name to initialize the model from scratch \n \
                                    4. Implement `_init_model` method.") 

    if _model.__class__ in MODEL_MAPPING.values():
        raise ValueError(f"Model class {_model.__class__} is MODEL_MAPPING and is not suitable modeling task")

    return _model
        
def create_optimizer_and_lr(ctx, optimizer_kwargs, lr_kwargs):
    """Create optimizer and learning rate scheduler for Trainer._init_setup method.
        Return default opt and lr if not provided"""
    # these setting are not necessary when using with Trainer and config, since config will handle all of empty optimizer_kwargs
    optimizer_kwargs["weight_decay"] = optimizer_kwargs.get("weight_decay", 0.0) 
    optimizer_kwargs["lr"] = ctx.config.learning_rate
    params_name = get_parameter_names(ctx.model) 
    decay_parameters = [p for p in params_name if not isinstance(p, non_decay_cls)] # filter e.g. LayerNorm
    decay_parameters = [p for p in params_name if "bias" not in p] # filter bias
    
    group_params = [
        {
            "params": [
                p for n, p in ctx.model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": optimizer_kwargs["weight_decay"], # For trainable params with weight decay
        },
        {
            "params": [
                p for n, p in ctx.model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,  # For trainable params without weight decay
        }
    ]
    optim = ctx.config.optimizer_cls(group_params, **optimizer_kwargs) # already assume that optimizer_cls_name is not None as this is handled by config
    lr_scheduler = ctx.config.scheduler_cls(optim, **lr_kwargs)
    
    return optim, lr_scheduler

def set_logger(rank=0, log_level=logging.INFO, log_dir="."): 
    
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "log.txt")),
            logging.StreamHandler()
        ]
    )
    logging.getLogger().setLevel(log_level)
    
    if rank == 0:
        datasets.utils.logging.set_verbosity_info()
        tfm.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        tfm.utils.logging.set_verbosity_error()
    
    torch.set_printoptions(precision=4, sci_mode=False)
    torch._logging.set_logs(all=log_level)

def log_on_main(*args, **kwargs):
    if D.get_rank() == 0:
        logging.getLogger().warning(*args, **kwargs)
        
def _pad(batch_list, max_length=1024, pading_value=0):
    assert isinstance(batch_list, (list, tuple)) and all([isinstance(x, (list, tuple)) for x in batch_list])

    if max_length is None:
        max_length = max([len(x) for x in batch_list])
    for i in range(len(batch_list)):
        batch_list[i] = batch_list[i] + [pading_value] * (max_length - len(batch_list[i]))
        
    return batch_list

def _valid_for_pad(x):
    '''check whether the input is type (tuple, list) with elements of type (tuple, list), and these elements are integers or floats'''
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], (list, tuple)): 
            return isinstance(x[0][0], (int, float)) # assume all elements are of the same type
    
def pad_across_structure(x, max_length=1024, pading_value=0):
    '''A padding function for any nested structure (e.g. dict of list of list). 
    If `max_length` is None, it will pad to maximum length in the batch.
    '''
    assert not isinstance(x, torch.Tensor), f"Not support padding across Tensor"
    if isinstance(x, dict):
        for key in x:
            x[key] = pad_across_structure(x[key], max_length=max_length, pading_value=pading_value)
        
    elif isinstance(x, (list, tuple)) and _valid_for_pad(x):
        return _pad(x, max_length=max_length, pading_value=pading_value)
    
    elif isinstance(x, (list, tuple)): # still list or tuple but not valid for padding
        for i in range(len(x)):
            x[i] = pad_across_structure(x[i], max_length=max_length, pading_value=pading_value)
    elif isinstance(x, (int, float)):
        raise ValueError(f"Cannot pad a single element {x}")
    else:
        raise ValueError(f"Unknown type {type(x)}")
    return x

def collate_fn(x):
    
    all_keys = x[0].keys() # assume all keys are the same
    x = {key: [x_i[key] for x_i in x] for key in all_keys}

    return x
        
def calculate_grad_penalty(loss, params, create_graph=True):
    
    grad_params = torch.autograd.grad(
        outputs=loss, inputs=params, create_graph=create_graph
    )
    norm = 0 
    for grad in grad_params:
        norm += grad.pow(2).sum()
    loss += norm.sqrt() 
    
    return loss 
