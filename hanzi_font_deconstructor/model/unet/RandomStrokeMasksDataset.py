import torch
from hanzi_font_deconstructor.common.generate_training_data import (
    get_training_input_and_mask_tensors,
)


class RandomStrokeMasksDataset(torch.utils.data.IterableDataset):
    def __init__(self, total_samples: int, size_px=512):
        super()
        self.total_samples = total_samples
        self.size_px = size_px

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        total_per_worker = int(self.total_samples / num_workers)
        for _ in range(total_per_worker):
            input, mask = get_training_input_and_mask_tensors(size_px=self.size_px)
            yield {
                "image": input,
                "mask": mask,
            }

    def __len__(self):
        return self.total_samples
