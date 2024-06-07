# Description: Distributed handler for multi-GPU training 
# (Only for Data Parallelism training scheme for single node, muti-node will be added later).
# (Other schemes will be support latter on)
import os 
import logging
from functools import partial
import contextlib
import dataclasses
import psutil
from functools import wraps
from typing import List, Tuple, Union, Generator
from collections.abc import Mapping 
import torch 
from torch import cuda
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP    
import torch.distributed as D

logger = logging.getLogger(__name__)

class DistributedException(Exception):
    pass

@dataclasses.dataclass
class DataStructure:
    """Data structure of the inputs."""
    shape: torch.Size
    dtype: torch.dtype

def _flatten_list(inputs) -> Generator:
    """Flatten the list or dict elements. (Do not preserve structure)"""
    if isinstance(inputs, (list, tuple)):
        for i in inputs:
            yield from _flatten_list(i)
    elif isinstance(inputs, Mapping):
        for v in inputs.values():
            yield from _flatten_list(v)
    else:
        yield inputs
    
def _recurse_func(func, inputs, *arg, test_type=torch.Tensor, **kwargs):
    """Recursively apply the function to the inputs where output structure is preserved."""
    if isinstance(inputs, Mapping):   
        return type(inputs)(
            {k : _recurse_func(func, v, *arg, **kwargs) for k, v in inputs.items()}
        )
    elif isinstance(inputs, (list, tuple)):
        return type(inputs)(_recurse_func(func, t, *arg, **kwargs) for t in inputs)
    elif isinstance(inputs, torch.Tensor):
        return func(inputs, *arg, **kwargs)
    elif isinstance(inputs, test_type):
        return func(inputs, *arg, **kwargs)
    else:
        raise TypeError(f"Unsupported type {type(inputs)} passed to {func.__name__}.")
    
def _put(tensor, device, dtype=None, non_blocking=True):
    
    try: # will try to convert as soon as possible
        return tensor.to(dtype=dtype, device=device, non_blocking=non_blocking)
    except: 
        pass 
    
    if isinstance(tensor, torch.Tensor) or hasattr(tensor, "to"):
        return tensor.to(dtype=dtype, device=device, non_blocking=non_blocking)
    elif isinstance(tensor, (list, tuple)):
        try: 
            tensor = torch.tensor(tensor, device=device)
            return tensor
        except: 
            return type(tensor)(
                _put(t, device, non_blocking=non_blocking) for t in tensor
            )
    elif isinstance(tensor, Mapping):
        return type(tensor)(
            {k: _put(v, device, non_blocking=non_blocking) for k, v in tensor.items()}
        )
    else:
        return tensor

def _get_data_structure(inputs):
    """Get the data structure of inputs (recursive)."""
    def _get_ops(tensor):
        return DataStructure(tensor.shape, tensor.dtype)
    return _recurse_func(_get_ops, inputs, test_type=DataStructure)

def _get_shape(inputs):
    """Get the shape of the inputs (recursive)."""
    def _get_ops(tensor):
        return list(tensor.shape)
    return _recurse_func(_get_ops, inputs, list)

def _slice(inputs, tensor_slice):
    """Slice the tensor inputs (recursive)."""
    def _slice_fn(tensor, tensor_slice):
        return tensor[tensor_slice]
    return _recurse_func(_slice_fn, inputs, tensor_slice)

def _find_device(inputs):
    """Get the device of the inputs, assuming all inputs share the same device. (return the first device found)"""
    if isinstance(inputs, torch.Tensor):
        return inputs.device
    if isinstance(inputs, (list, tuple)):
        for i in inputs:
            return _find_device(i)
    if isinstance(inputs, dict):
        for v in inputs.values():
            return _find_device(v)
    raise ValueError("Unsupported input type.")

def _get_shape(inputs):
    """Get the shape of the inputs (recursive)."""
    def _get_ops(tensor):
        return list(tensor.shape)
    return _recurse_func(_get_ops, inputs)

def _gather_obj(obj):
    output_objects = [None] * D.get_world_size() 
    D.all_gather_object(output_objects, obj)
    return output_objects

def verify_ops(func):
    """Verify the that inputs tensor to function share the same data structure across all devices."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        inputs = kwargs.get("tensor", args[0])
        shapes = _get_shape(inputs)
        gathered_shapes = _gather_obj([shapes])
        if gathered_shapes[0] is not None and gathered_shapes.count(gathered_shapes[0]) != len(gathered_shapes):
            shape_idx =  "\n  - ".join([f"Process {i}: {shape}" for i, shape in enumerate(gathered_shapes)])
            raise DistributedException(
                f"Data structure mismatch in `{func.__module__}.{func.__name__}`:\n  - {shape_idx}"
            )
        return func(*args, **kwargs)
    return wrapper


def chain_ops(func):
    """Check for errors in the function and raise DistributedException."""
    @wraps(func)
    def wrapper(self, *args, **kwargs): # similar to accelerate 
        try:
            return func( *args, **kwargs)
        except DistributedException as e:
            raise DistributedException(f"Error in `{func.__module__}.{func.__name__}`: {e}")
    return wrapper

@verify_ops
def gather(tensor):
    """Synchronous all_gather of tensor across all processes. (recursive torch.distributed.all_gather_into_tensor)"""
    # if D.get_world_size() == 1:
    #     return tensor
    def _gather_fn(tensor):
        if tensor.ndim == 0:
            tensor = tensor.clone()[None] # use instead of tensor.unsqueeze(0)
        
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        empty = torch.empty(D.get_world_size() * tensor.numel(), dtype=tensor.dtype, device=tensor.device)
        D.all_gather_into_tensor(empty, tensor)
        return empty.view(-1, *tensor.shape[1:]) # from shape of (world_size, ...)
    
    return _recurse_func(_gather_fn, tensor)
    
@verify_ops
def reduce(tensor, reduction=D.ReduceOp.AVG):
    """Reduce across all processes. (recursive torch.distributed.all_reduce)"""
    # if D.get_world_size() == 1:
    #     return tensor
    tensor = tensor.clone()
    _reduce = lambda x: D.all_reduce(x, reduction=reduction)
    return _recurse_func(_reduce, tensor)

@contextlib.contextmanager
def split_across_process(tensor):
    """Split across processes. The batch_size should be evenly divisible by the world_size, otherwise this will throw an error."""
    num_proc = D.get_world_size()
    rank = D.get_rank()
    
    if num_proc == 1:
        yield tensor
        return # exit the context manager
    
    if isinstance(tensor, dict):
        if not all(len(v) == len(next(iter(tensor.values()))) for v in tensor.values()):
            raise ValueError("All values in the dictionary must have the same length.")
    num_sample_per_proc = len(tensor) // num_proc 
    if num_sample_per_proc * num_proc != len(tensor):
        raise ValueError("The number of samples must be evenly divisible by the world_size.")
    
    start = rank * num_sample_per_proc
    end = start + num_sample_per_proc
    def _recurse(tensor, start, end):
        
        if isinstance(tensor, dict):
            for key in tensor.keys():
                tensor[key] = _recurse(tensor[key], start, end)
            return tensor

        elif isinstance(tensor, (list, tuple, torch.Tensor)):
            if start >= len(tensor):
                return tensor[-1:] 
            return tensor[start:end]
        else:
            raise ValueError("Input type not supported.")
    
    yield _recurse(tensor, start, end)
    
def init_worker(rank, world_size, func=None, arg=()):

    os.environ["MASTER_ADDR"] = "localhost"
    #os.environ["MASTER_PORT"] = "3035" 
    master_port = os.environ["MASTER_PORT"]
    backend = D.Backend.NCCL if cuda.is_available() else D.Backend.GLOO
    
    if not D.is_initialized():
        D.init_process_group(backend=backend, world_size=world_size, rank=rank)
    logger.warning(f"[{os.getpid()}] PORT: {master_port} Initialized process group: rank={D.get_rank()}, world_size={D.get_world_size()}")

def gather_params(model):
    """Gather model parameters across all processes. (DDP only), ensuring that all processes have the same model."""
    
def pad_across_process(tensor):
    pass 




