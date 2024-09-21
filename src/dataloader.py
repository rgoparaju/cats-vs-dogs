from torch.utils.data import DataLoader
from src.dataset import DogsVsCatsDataset
from torch.utils.data.dataloader import default_collate


class DogsVsCatsDataloader(DataLoader):

    def __init__(self, data: DogsVsCatsDataset, batch_size: int, shuffle: bool):
        super().__init__(dataset = data, batch_size = batch_size, shuffle = shuffle)

    def collate_fn(self, batch):
        print(batch is None)
        new_batch = []
        for image, label in batch:
            if image != None:
                new_batch.append((image, label))
        return default_collate(new_batch)