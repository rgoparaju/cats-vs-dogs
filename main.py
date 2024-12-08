from typing import Tuple

from src.img_annotator import PetImgAnnotator
from src.dataset import DogsVsCatsDataset

import torch
from torch.utils.data import DataLoader

from src.net import Net
from torch.nn import BCELoss
import torch.optim as optim

from src.collate import custom_collate


def get_annotations(
        seed: int,
        cats_src: str,
        dogs_src: str,
        train_split: float
) -> Tuple[str]:
    """
        given a random seed to split the data into train/test,
        and source dirs for the cat and dog images, initializes the
        PetImgAnnotator class and calls the create_pet_img_annotations
        function.

        Args:
            - seed: random seed to split into train.test
            - cats_src: source folder for all cat images
            - dogs_src: source folder for all dog images

        Returns:
            - the filenames of the annotation .csv files
            in a tuple
    """
    annotator = PetImgAnnotator(
        seed,
        cats_src,
        dogs_src
    )
    train_labels, test_labels = annotator.create_pet_img_annotations(train_split)

    return train_labels, test_labels

def train(
        n_epochs: int,
        traindata: DogsVsCatsDataset,
        bs: int
) -> Tuple[str]:
    """
        Initialize a new CNN with learning rate 0.001
        and momentum 0.9.

        Train the CNN for the given epochs on the dataset
        and save the weights.
        Returns a tuple containing the strings corresponding
        to the filename of the train loss per epoch for plotting,
        and the filename of the saved model weights.
    """

    train_dataloader = DataLoader(
        traindata,
        batch_size = bs,
        shuffle = True,
        num_workers = 0,
        collate_fn = custom_collate
    )

    net = Net()
    loss_fn = BCELoss()
    optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)

    for _ in range(n_epochs):
        for inputs, labels in train_dataloader:
            preds = net(inputs)
            optimizer.zero_grad()
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()

    weights_file = "saved_weights.pth"
    torch.save(net.state_dict(), weights_file)
    return weights_file

def test(
        saved_weights: str,
        testdata: DogsVsCatsDataset,
        bs: int
):
    net = Net()
    net.load_state_dict(torch.load(saved_weights))

    testdataloader = DataLoader(
        testdata,
        batch_size = bs,
        shuffle = True,
        num_workers = 0,
        collate_fn = custom_collate
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testdataloader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels[:, 0]).sum().item()
    print(f"Accuracy: {100 * correct // total}%")

def main():
    seed = 0
    # cats_src = r"data\PetImages\Cat"
    # dogs_src = r"data\PetImages\Dog"
    cats_src = r"C:\Users\Ravi\Documents\Python Projects\cats_vs_dogs old\data\PetImages\Cat"
    dogs_src = r"C:\Users\Ravi\Documents\Python Projects\cats_vs_dogs old\data\PetImages\Dog"
    train_split = 0.8
    bs = 16
    n_epochs = 10

    train_labels, test_labels = get_annotations(
        seed=seed,
        cats_src=cats_src,
        dogs_src=dogs_src,
        train_split=train_split
    )

    traindata = DogsVsCatsDataset(train_labels)
    testdata = DogsVsCatsDataset(test_labels)

    ### training loop
    saved_weights = train(n_epochs, traindata, bs)

    ### testing
    test(saved_weights, testdata, bs)


if __name__ == "__main__":
    main()