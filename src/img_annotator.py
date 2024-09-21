import os
import numpy as np
from random import seed, shuffle
from typing import Tuple, List

class PetImgAnnotator:
    """
        class for creating annotation files for the
        training and testing sets for the Dogs Vs Cats
        image dataset
    """
    def __init__(
            self,
            seed: int,
            cats_folder: str,
            dogs_folder: str
    ) -> None:
        """
            args:
                - seed: random seed used for splitting data into train and test
                - cats_folder: root dir for the cat imgs
                - dogs_folder: root dir for the dog imgs
        """
        self.seed = seed

        assert os.path.exists(cats_folder), "cat images source directory not found"
        assert os.path.exists(dogs_folder), "dog images source directory not found"
        self.cats_folder = cats_folder
        self.dogs_folder = dogs_folder

    def filter_non_imgs(
            self,
            imgs_list: List
    ) -> List[str]:
        return list(
            filter(
                lambda filename: "jpg" in filename,
                imgs_list
            )
        )
    
    def get_split(
            self,
            train_split: float = 0.8
    ) -> Tuple[List[str], List[str]]:
        """
            given the train split passed from create_pet_img_annotations,
            randomly partition the image data into training and testing
            sets and return them

            args:
                - train_split: number between 0 and 1 which determines 
                    how much of the data is set aside for testing

            returns:
                - train: list of filenames of training imgs
                - test: list of filenames of testing imgs
        """
        # set the random seed
        seed(self.seed)

        # get the lists of cat images and dog images,
        # only keeping filenames that have "jpg" in them
        cat_imgs_list = os.listdir(self.cats_folder)
        cats_imgs = self.filter_non_imgs(cat_imgs_list)
        # cats_imgs = [
        #     os.path.join(self.cats_folder, img)
        #     for img in cat_imgs_list
        #     if "jpg" in img
        #     ]
        
        dog_imgs_list = os.listdir(self.dogs_folder)
        dogs_imgs = self.filter_non_imgs(dog_imgs_list)
        # dogs_imgs = [
        #     os.path.join(self.dogs_folder, img)
        #     for img in dog_imgs_list
        #     if "jpg" in img
        #     ]

        # combine the two datasets and shuffle
        total = cats_imgs + dogs_imgs
        shuffle(total)

        # split the data by the train_split passed in
        split_index = int(np.floor(len(total) * train_split))
        train_imgs = total[:split_index]
        test_imgs = total[split_index:]

        return train_imgs, test_imgs

    def create_pet_img_annotations(
            self,
            train_split: float = 0.8,
            train_output_path: str = "dogs_cats_train_labels.csv",
            test_output_path: str = "dogs_cats_test_labels.csv"
    ) -> Tuple[str, str]:
        """
            function to generate the csv files of labels for the image data.
            
            Args:
                - train_split: number between 0 and 1 which determines 
                how much of the data is set aside for testing
            Returns:
                - Tuple of two csv file names,
                train_output_path and test_output_path
        """
        if train_split < 0 or train_split > 1:
            raise ValueError("train_split must be between 0 and 1")
        train_imgs, test_imgs = self.get_split(train_split)

        with open(train_output_path, "w") as f:
            for trn_img in train_imgs:
                label = "1,0" if "Cat" in trn_img else "0,1"
                f.write(f"{trn_img},{label}\n")

        with open(test_output_path, "w") as f:
            for tst_img in test_imgs:
                label = "1,0" if "Cat" in tst_img else "0,1"
                f.write(f"{tst_img},{label}\n")

        return train_output_path, test_output_path
