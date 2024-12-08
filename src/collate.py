import torch
from torch import Tensor
from torch.utils.data.dataloader import default_collate

def custom_collate(batch: Tensor) -> Tensor:
    # batch = list(filter(lambda x:x is not None, batch))
    new_batch = []
    for image, label in batch:
        if image is not None:
            new_batch.append((image, label))
    if not new_batch:
        return torch.empty((0,))

    return default_collate(new_batch)