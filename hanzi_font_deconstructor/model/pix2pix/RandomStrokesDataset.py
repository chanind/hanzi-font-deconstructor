import torch
from hanzi_font_deconstructor.common.generate_training_data import (
    get_training_input_and_mask_tensors,
)


class RandomStrokesDataset(torch.utils.data.IterableDataset):
    def __init__(self, total_samples: int, size_px=512):
        super()
        self.total_samples = total_samples
        self.size_px = size_px

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        total_per_worker = int(self.total_samples / num_workers)
        for _ in range(total_per_worker):
            with torch.no_grad():
                input, mask = get_training_input_and_mask_tensors(size_px=self.size_px)
                # convert to float, scaled between 0 - 1 (2 is the overlap mask class)
                # unsqueeze to make a 1-channel image
                target = mask.unsqueeze(0) / 2
            yield input, target

    def __len__(self):
        return self.total_samples
