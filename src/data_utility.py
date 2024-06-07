
import logging 
import inspect
from typing import Iterable, List, Dict, Any, Optional
import math
import heapq

import torch
from torch.utils.data import DataLoader, Sampler
import torch.distributed as D
from dist import _put
# Note that `constant_length_dataset` is generally preferred over `constant_length_dataset_with_sort`
# as it will produce less example and number of padding # mostly will reduce number of example 

def get_signature_col(model):
    """Get the signature columns of the model forward method."""
    signature = inspect.signature(model.forward)
    _signature_columns = list(signature.parameters.keys())
    if _signature_columns[0] == "self":
        _signature_columns = _signature_columns[1:]
    return _signature_columns

def remove_unused_col(model, dataset):
    """Remove columns from the dataset that are not used in the model forward method."""
    signature_columns = get_signature_col(model)
    ignored_col = set(dataset.column_names) - set(signature_columns)
    dataset = dataset.remove_columns(ignored_col)
    return dataset, ignored_col

def partition_approx_ele(
    arr: List[int], K: int
) -> List[List[int]]:
    """
    Partitions an array into subarrays such that the sum of the elements in each subarray is as close as possible to K.
    
    Args:
        arr (List[int]): The input array of integers.
        K (int): The target sum for each subarray.
        
    Returns:
        List[List[int]]: A list of subarrays (each represented by a list of indices) where the sum of the elements in
        each subarray is as close as possible to K.
    """
    # Convert elements > K to their remainder when divided by K
    arr_mod = [num % K for num in arr]
    indexed_arr = [(idx, arr[idx], mod_value) for idx, mod_value in enumerate(arr_mod)]
    indexed_arr.sort(key=lambda x: x[2], reverse=True)

    min_heap = []
    for idx, original_value, mod_value in indexed_arr:
        if min_heap and (min_heap[0][0] + mod_value <= K):
            curr_sum, sublist = heapq.heappop(min_heap)
            sublist.append(idx)
            current_sum = curr_sum + mod_value
            heapq.heappush(min_heap, (current_sum, sublist))
        else:
            new_sum = mod_value
            new_sublist = [idx]
            heapq.heappush(min_heap, (new_sum, new_sublist))

    # Retrieve results from the heap and sort by the sum of their actual values
    result = [sublist for _, sublist in min_heap]
    result.sort(key=lambda sublist: sum(arr[i] for i in sublist))

    return result

def constant_length_dataset(
        x: Dict[Any, List[List[int]]], seq_length: int = 1024, 
        threshold: int = 10, eos_token_id: int = 11, 
        pad_token_id: int = -100
) -> Dict[Any, List[List[int]]]:
    """
    This function maps a dictionary of lists of integers into a dictionary of fixed-length sequences.
    Args:
    x (Dict[Any, List[List[int]]]): The input dictionary where the values are lists of lists of integers.
    seq_length (int): The desired sequence length.
    threshold (int): If math.fabs(len(x[i]) - seq_length) < threshold, it will be padded to seq_length, 
                     otherwise will be normally concatenated with other batches.
    eos_token_id (int): The token id that will be added to the end of each batch.
    pad_token_id (int): The token id that will be used to pad the new batch for non-concatenated batch.
    
    Returns:
    Dict[Any, List[List[int]]]: A dictionary of fixed-length sequences.
    """
    results = {}

    for key, value in x.items():
        results[key] = []
        block = []

        # Initialize lists to separate batches that need padding and those to be concatenated
        pad_batches = []
        concat_batches = []

        for batch in value:
            if math.fabs(len(batch) - seq_length) <= threshold:
                pad_batches.append(batch)
            else:
                concat_batches.append(batch)

        # Process padded batches
        for batch in pad_batches:
            padded_batch = batch + [pad_token_id] * (seq_length - len(batch))
            results[key].append(padded_batch)

        # Concatenate remaining batches
        for batch in concat_batches:
            block.extend(batch + [eos_token_id])
            while len(block) >= seq_length:
                results[key].append(block[:seq_length])
                block = block[seq_length:]

        # If there is still some element left
        if len(block) > 0:
            results[key].append(block + [pad_token_id] * (seq_length - len(block)))

    return results

def constant_length_dataset_with_sort_(
    arr: List[List[int]], seq_length: int = 1024, eos_token_id: int = -10, pad_token_id: int = -100
) -> List[List[int]]:
    """
    This function maps a list of lists of integers into a list of fixed-length sequences.
    have similar usage as `constant_length_dataset`, but it will use `partition_approx_ele` to partition the sequence
    and map the indices to the original list.
    """
    seq_lens = list(map(len, arr))
    # get the partition
    partitions = partition_approx_ele(seq_lens, seq_length)
    results = []
    
    for partition in partitions:
        block = []
        for idx in partition:
            block.extend(arr[idx] + [eos_token_id])
            
        while len(block) >= seq_length:
            results.append(block[:seq_length])
            block = block[seq_length:]
        if len(block) > 0: # maybe not consider this part at all since it is negligible (as for `constant_length_dataset``)
            results.append(block + [pad_token_id] * (seq_length - len(block)))
            
    return results

def constant_length_dataset_with_sort(
    x: Dict[Any, List[List[int]]], seq_length: int = 1024, eos_token_id: int = -10, pad_token_id: int = -100
):
    """This function maps a dictionary of lists of integers into a dictionary of fixed-length sequences."""
    results = {}
    for key, value in x.items():
        results[key] = constant_length_dataset_with_sort_(value, seq_length, eos_token_id, pad_token_id)

    return results

class DistributedGroupedSampler(Sampler): #TODO: implement a more fine-grained version of this sampler
    """
        Sampler that groups dataset indices by similar sequence lengths and distributes them
        across multiple processes for distributed training. This sampler is useful for minimizing padding when training language models by grouping
        sequences of similar lengths together.
    """
    def __init__(
        self, 
        dataset, 
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        num_multi_batch: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        model_input_name: str = None
    ) -> None:
        if num_replicas is None:
            if not D.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = D.get_world_size()
        if rank is None:
            if not D.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = D.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.feature = list(self.dataset.features)[0] if model_input_name is None else model_input_name
        self.dataset_length = len(dataset)
        
        if num_multi_batch is None:
            num_multi_batch = max(self.dataset_length // 1000, 1)
        
        self.num_multi_batch = num_multi_batch

        # Determine the number of samples and total size
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # Grouping logic from GroupedSampler
        indices = torch.randperm(self.dataset_length, generator=g).tolist()
        grouped_indices = [
            indices[i: i + self.num_multi_batch]
            for i in range(0, self.dataset_length, self.num_multi_batch)
        ]
        grouped_indices = [
            sorted(batch, key=lambda i: len(self.dataset[i][self.feature]), reverse=True)
            for batch in grouped_indices
        ]
        flat_indices = [i for m in grouped_indices for i in m]

        # Distributed logic from DistributedSampler
        if not self.drop_last: # note: in Training Script, we always automatically drop the Dataset.
            padding_size = self.total_size - len(flat_indices)
            if padding_size <= len(flat_indices):
                flat_indices += flat_indices[:padding_size]
            else:
                flat_indices += (flat_indices * math.ceil(padding_size / len(flat_indices)))[:padding_size]
        else:
            flat_indices = flat_indices[:self.total_size]

        assert len(flat_indices) == self.total_size

        # Subsample for the current process
        subsampled_indices = flat_indices[self.rank:self.total_size:self.num_replicas]
        assert len(subsampled_indices) == self.num_samples

        return iter(subsampled_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        
def collate_func(x):
    # passed in being list of dict of size batch_size
    all_keys = x[0].keys() # assume all keys are the same
    x = {key: [x_i[key] for x_i in x] for key in all_keys}
    
    return x 