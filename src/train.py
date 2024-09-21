from typing import Tuple

from src.img_annotator import PetImgAnnotator
from src.dataset import DogsVsCatsDataset
# from dataloader import DogsVsCatsDataloader
import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
# import torchvision
from src.net import Net
from torch.nn import BCELoss
import torch.optim as optim
# from tqdm import tqdm


def custom_collate(batch):
    # batch = list(filter(lambda x:x is not None, batch))
    new_batch = []
    for image, label in batch:
        if image != None:
            new_batch.append((image, label))
    return default_collate(new_batch)


def get_annotations(
        seed: int,
        cats_src: str,
        dogs_src: str,
        overwrite: bool = True
) -> Tuple[str]:
    """
        given a random seed to split the data into train/test,
        and source dirs for the cat and dog images, initializes the
        PetImgAnnotator class and calls the create_pet_img_annotations
        function. if the overwrite arg is True, then the function first
        checks to see if annotations already exist before calling the 
        annotation function.

        Args:
            - seed: random seed to split into train.test
            - cats_src: source folder for all cat images
            - dogs_src: source folder for all dog images
            - overwrite: determines whether any new 
            annotations files are created to overwrite 
            the existing ones.

        Returns:
            - the filenames of the annotation .csv files
            in a tuple
    """
    annotator = PetImgAnnotator(
        0,
        cats_src,
        dogs_src
    )
    train_labels, test_labels = annotator.create_pet_img_annotations(

    )

    return train_labels, test_labels


if __name__ == "__main__":
    cats_src = r"data\PetImages\Cat"
    dogs_src = r"data\PetImages\Dog"
    train_split = 0.8

    train_labels, test_labels = get_annotations(
        cats_src=cats_src,
        dogs_src=dogs_src
    )