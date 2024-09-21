import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import *
# from torchvision.transforms.functional import to_grayscale


class DogsVsCatsDataset(Dataset):
    """
        create a custom dataset class for the cat and dog images from an
        annotations .csv file. if this file does not exist,
        initializes an image transform pipeline to apply.
    """
    def __init__(self, annotations_file):
        self.imgs_labels = pd.read_csv(annotations_file, header = None)
        self.transforms = Compose([
            ToTensor(),
            Resize((32, 32))
            #  ,Grayscale()
        ])

    def __len__(self):
        return len(self.imgs_labels)

    def __getitem__(self, index):
        """
            
        """
        path = self.imgs_labels.iloc[index, 0]
        try:
            image = Image.open(path)
            image = self.transforms(image)
            # image = self.transforms(to_grayscale(image)) # workaround to color images that are somehow loading in as grayscale by PIL (idk why it happens)
            if image.shape[0] != 3: # checks if the first dimension (color channel) of the loaded image is not 3, returns none if so
                return None, None

            label = np.array(self.imgs_labels.iloc[index, 1:])
            label = label.astype("float32")
            return image, label

        except:
            return None, None
