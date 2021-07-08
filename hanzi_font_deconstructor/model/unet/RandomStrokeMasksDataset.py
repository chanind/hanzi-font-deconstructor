import torch
from hanzi_font_deconstructor.common.generate_training_data import (
    get_training_input_and_mask_tensors,
)


class RandomStrokeMasksDataset(torch.utils.data.IterableDataset):
    def __init__(self, total_samples: int, size_px=512, static=False):
        """
        static means to pregenerate all samples up front for consistency, rather than doing it on the fly
        """
        super()
        self.total_samples = total_samples
        self.size_px = size_px
        self.pregenerated_samples = None
        if static:
            self.pregenerated_samples = [
                self.generate_sample() for _ in range(total_samples)
            ]

    def generate_sample(self):
        input, mask = get_training_input_and_mask_tensors(size_px=self.size_px)
        return {
            "image": input,
            "mask": mask,
        }

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        total_per_worker = int(self.total_samples / num_workers)
        for i in range(total_per_worker):
            if self.pregenerated_samples:
                yield self.pregenerated_samples[i]
            else:
                yield self.generate_sample()

    def __len__(self):
        return self.total_samples
