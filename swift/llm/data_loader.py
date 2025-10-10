from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from swift.llm import to_device


class BatchSamplerShard:

    def __init__(self,
                 total_samples: int,
                 batch_size: int,
                 shuffle: bool,
                 drop_last: bool,
                 data_seed: Optional[int],
                 tp_size: int = 1):
        self.tp_size = tp_size
        self.total_samples = total_samples // self.world_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.base_seed = data_seed or 0
        self.curr_seed = self.base_seed

    @property
    def rank(self):
        return (dist.get_rank() // self.tp_size) if dist.is_initialized() else 0

    @property
    def world_size(self):
        return (dist.get_world_size() // self.tp_size) if dist.is_initialized() else 1

    def __iter__(self):
        start_idx = self.rank * self.total_samples
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.curr_seed)
            total_idx = torch.randperm(self.total_samples * self.world_size, generator=generator).tolist()
            total_idx = total_idx[start_idx:start_idx + self.total_samples]
        else:
            total_idx = list(range(start_idx, start_idx + self.total_samples))

        batch = []
        # Last batch if not complete will be dropped.
        for idx in total_idx:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if not self.drop_last and len(batch) > 0:
            yield batch
        return

    def set_epoch(self, epoch: int):
        self.curr_seed = self.base_seed + epoch

    def __len__(self) -> int:
        if self.drop_last:
            return self.total_samples // self.batch_size
        else:
            return (self.total_samples + self.batch_size - 1) // self.batch_size


class DataLoaderShard(DataLoader):

    def __init__(self, dataset, device=None, **dataloader_params):
        self.device = device
        super().__init__(dataset, **dataloader_params)

    def set_epoch(self, epoch: int):
        if self.batch_sampler is not None and hasattr(self.batch_sampler, 'set_epoch'):
            self.batch_sampler.set_epoch(epoch)
        elif self.sampler is not None and hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)

    def __iter__(self):
        for item in super().__iter__():
            if self.device:
                item = to_device(item, self.device)
            yield item


class ZoneBasedBatchSampler(BatchSamplerShard):
    """Zone-based progressive sampler for curriculum learning.
    
    Divides dataset into N zones. At epoch i, only samples from zone i are used.
    Each zone maintains internal shuffling for better generalization.
    
    Example:
        - Dataset: 5000 samples, num_zones=5, num_epochs=5
        - Zone 0 (samples 0-999):    Used in epoch 0
        - Zone 1 (samples 1000-1999): Used in epoch 1
        - Zone 2 (samples 2000-2999): Used in epoch 2
        - And so on...
    """

    def __init__(self,
                 total_samples: int,
                 batch_size: int,
                 shuffle: bool,
                 drop_last: bool,
                 data_seed: Optional[int],
                 num_zones: int,
                 tp_size: int = 1):
        """
        Args:
            total_samples: Total number of samples across all zones
            batch_size: Batch size
            shuffle: Whether to shuffle within each zone
            drop_last: Whether to drop last incomplete batch
            data_seed: Random seed for shuffling
            num_zones: Number of zones (typically equals num_epochs)
            tp_size: Tensor parallel size
        """
        # Don't call super().__init__ yet, we need to override total_samples
        self.tp_size = tp_size
        self.num_zones = num_zones
        self.zone_size = total_samples // num_zones
        self.original_total_samples = total_samples
        self.current_epoch = 0
        
        # Validate zone division
        assert total_samples % num_zones == 0, \
            f"total_samples ({total_samples}) must be divisible by num_zones ({num_zones})"
        
        # Initialize parent class with zone_size as total_samples
        # Each epoch only uses one zone
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.base_seed = data_seed or 0
        self.curr_seed = self.base_seed
        
        # Override total_samples to be zone_size per rank
        self.total_samples = self.zone_size // self.world_size

    def __iter__(self):
        # Calculate which zone to use based on current epoch
        zone_id = self.current_epoch % self.num_zones
        zone_start = zone_id * self.zone_size
        zone_end = zone_start + self.zone_size
        
        # Print zone switching information (for debug)
        print(f"Switching data to Zone {zone_id} (samples {zone_start}-{zone_end-1}) for epoch {self.current_epoch}")
        
        # Adjust for distributed training
        samples_per_rank = self.zone_size // self.world_size
        rank_start_in_zone = self.rank * samples_per_rank
        
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.curr_seed)
            # Only shuffle within the current zone
            zone_indices = torch.randperm(self.zone_size, generator=generator).tolist()
            # Offset indices to the actual zone position in dataset
            zone_indices = [idx + zone_start for idx in zone_indices]
            # Get this rank's portion
            total_idx = zone_indices[rank_start_in_zone:rank_start_in_zone + samples_per_rank]
        else:
            # Sequential sampling within zone
            total_idx = list(range(zone_start + rank_start_in_zone, 
                                   zone_start + rank_start_in_zone + samples_per_rank))
        
        # Batch the indices
        batch = []
        for idx in total_idx:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if not self.drop_last and len(batch) > 0:
            yield batch

    def set_epoch(self, epoch: int):
        """Update epoch for both zone selection and shuffling."""
        self.current_epoch = epoch
        self.curr_seed = self.base_seed + epoch

    def __len__(self) -> int:
        # Length is based on zone size, not total samples
        samples_per_rank = self.zone_size // self.world_size
        if self.drop_last:
            return samples_per_rank // self.batch_size
        else:
            return (samples_per_rank + self.batch_size - 1) // self.batch_size


class DataLoaderDispatcher:

    def __init__(self, base_dataloader, device=None, skip_batches: int = 0):
        self.base_dataloader = base_dataloader
        self.device = device
        self.skip_batches = skip_batches

    @property
    def rank(self):
        return dist.get_rank(self.group) if dist.is_initialized() else 0

    @property
    def world_size(self):
        return dist.get_world_size(self.group) if dist.is_initialized() else 1

    @property
    def group(self):
        return dist.group.WORLD if dist.is_initialized() else 1

    def _scatter_object_list(self, inputs):
        if not dist.is_initialized():
            return inputs[0]
        outputs = [None]
        global_src_rank = dist.get_global_rank(self.group, 0)
        dist.scatter_object_list(outputs, inputs, global_src_rank, group=self.group)
        return outputs[0]

    def _skip_batches(self, base_iter):
        if self.rank == 0 and self.skip_batches > 0:
            for _ in tqdm(range(self.skip_batches), dynamic_ncols=True, desc='Skip Batches: '):
                [next(base_iter) for _ in range(self.world_size)]

    def __iter__(self):
        base_iter = iter(self.base_dataloader)
        self._skip_batches(base_iter)
        while True:
            if self.rank == 0:
                try:
                    data = [next(base_iter) for _ in range(self.world_size)]
                except StopIteration:
                    data = [None] * self.world_size
                data = self._scatter_object_list(data)
            else:
                data = self._scatter_object_list(None)
            if data is None:
                break
            if self.device:
                data = to_device(data, self.device)
            yield data
