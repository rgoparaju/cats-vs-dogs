from torch import Tensor
from torch.utils.data.dataloader import default_collate

def custom_collate(batch: Tensor) -> Tensor:
    # batch = list(filter(lambda x:x is not None, batch))
    new_batch = []
    for image, label in batch:
        if image != None:
            new_batch.append((image, label))
    return default_collate(new_batch)