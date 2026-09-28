import time

import torch

from lavps.utils.experiment import clean_cache

BYTES_PER_MEGABYTE = 1000 ** 2


class ExecutionProfiling:
    """Context manager that measures time and GPU usage for a block of code."""
    def __init__(self, batch_size: int = 1, num_iterations: int = 1, print_information: bool = True):
        self.stats = {}
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.print_information = print_information

    def __enter__(self):
        """On entering the context, measure 'pre' stats."""

        # All things considered it seems better to clean before
        clean_cache()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Note:
        # torch.cuda.memory_allocated()
        # GPU Memory used by all active tensors (will grow/shrink as you allocate/free).
        # torch.cuda.memory_reserved()
        # Total GPU memory reserved by PyTorch ~ nvidia-smi

        # Record the state of the memory prior to the sampling
        self.stats["memory_allocated_pre_sampling"] = torch.cuda.memory_allocated() / BYTES_PER_MEGABYTE
        self.stats["memory_reserved_pre_sampling"] = torch.cuda.memory_reserved() / BYTES_PER_MEGABYTE

        if self.print_information:
            # print(f'memory_allocated_pre_sampling: {self.stats["memory_allocated_pre_sampling"]}')
            print(f'memory_reserved_pre_sampling: {self.stats["memory_reserved_pre_sampling"]}')

        self.start_time = time.perf_counter()
        return self.stats

    def __exit__(self, exc_type, exc_val, exc_tb):
        """On exiting the context, measure 'post' stats and finalize dictionary."""

        end_time = time.perf_counter()
        total_time = end_time - self.start_time

        self.stats["total_time"] = total_time
        self.stats["time_per_image"] = total_time / self.batch_size
        self.stats["time_per_iteration"] = total_time / self.num_iterations
        self.stats["time_per_iteration_per_image"] = total_time / (self.num_iterations * self.batch_size)

        # The memory that we will show on the different graphs will be this one: max_memory_reserved_during_sampling
        # (the maximum memory reserved during sampling)
        self.stats["max_memory_allocated_during_sampling"] = torch.cuda.max_memory_allocated() / BYTES_PER_MEGABYTE
        self.stats["max_memory_reserved_during_sampling"] = torch.cuda.max_memory_reserved() / BYTES_PER_MEGABYTE

        # Those are useful to know exactly how much memory the sampling process uses
        self.stats["memory_allocated_for_sampling"] = self.stats["max_memory_allocated_during_sampling"] - self.stats["memory_allocated_pre_sampling"]
        self.stats["memory_allocated_for_sampling_per_sample"] = self.stats["memory_allocated_for_sampling"]/self.batch_size

        self.stats["memory_reserved_for_sampling"] = self.stats["max_memory_reserved_during_sampling"] - self.stats["memory_reserved_pre_sampling"]
        self.stats["memory_reserved_for_sampling_per_sample"] = self.stats["memory_reserved_for_sampling"]/self.batch_size

        clean_cache()

        # Those are to know the leakage
        self.stats["memory_allocated_post_sampling_post_cleaning"] = torch.cuda.memory_allocated() / BYTES_PER_MEGABYTE
        self.stats["memory_allocated_post_sampling_leaked"] = self.stats["memory_allocated_post_sampling_post_cleaning"] - self.stats["memory_allocated_pre_sampling"]

        self.stats["memory_reserved_post_sampling_post_cleaning"] = torch.cuda.memory_reserved() / BYTES_PER_MEGABYTE
        self.stats["memory_reserved_post_sampling_leaked"] = self.stats["memory_reserved_post_sampling_post_cleaning"] - self.stats["memory_reserved_pre_sampling"]

        if self.print_information:
            # print(f'max_memory_allocated_during_sampling: {self.stats["max_memory_allocated_during_sampling"]}')
            print(f'max_memory_reserved_during_sampling: {self.stats["max_memory_reserved_during_sampling"]}')
            # print(f'memory_allocated_post_sampling_post_cleaning: {self.stats["memory_allocated_post_sampling_post_cleaning"]}')
            # print(f'memory_reserved_post_sampling_post_cleaning: {self.stats["memory_reserved_post_sampling_post_cleaning"]}')
