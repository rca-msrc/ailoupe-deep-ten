import os
import os.path
import itertools
from PIL import Image

import torch
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets, models, transforms


import pandas as pd
import random
import pathlib

_imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}

class SingleImageDataset(Dataset):
    """SingleImageDataset """
    def __init__(self, index_path, split=None, classes=None, n_per_class=None, transform=None):
        self.index_df = pd.read_csv(index_path)
        self.transform = transform
        self.n_per_class = n_per_class

        if split is not None:
            self.index_df = self.index_df[self.index_df.split == split]
        
        if classes is None:
            self.classes = sorted(self.index_df.class_name.unique())
        else:
            self.classes = classes
    
        # calculate labels from classes
        self.index_df["label"] = self.index_df.class_name.map(lambda c: self.classes.index(c))
        
        # per group
        self.image_paths = []
        self.labels = []
        for label, group in self.index_df.groupby("label"):
            class_image_paths = list(group.image_path)

            if self.n_per_class is not None: # limit to a random n images per class
                class_image_paths = random.sample(class_image_paths, min(len(class_image_paths), self.n_per_class))
                            
            # add to the dataset
            self.image_paths.extend(class_image_paths)
            self.labels.extend([label] * len(class_image_paths))
            # print(label, len(group), len(class_image_paths), len(image_paths), 
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):        
        image = Image.open(self.image_paths[idx]).convert("RGB")
        name = self.image_paths[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, name

class DualImageDataset(Dataset):
    """DualImageDataset - A and B images paired
        - pairing strategy:
            "product": zip pair size, cross-product, combination without replacement,
            "session": use the "session" column to pair A and B images
        - inverse: A,B include B,A as inputs (doubles the size), same set of images (for product)
        - n_per_class: limit number of examples per class
        - random_sample: randomise if there are more than n_per_class
    """
    def __init__(self, index_path, split=None, classes=None, n_per_class=None, pairing_strategy="product", include_inverse=True, transform=None):
        self.index_df = pd.read_csv(index_path)
        if split is not None:
            self.index_df = self.index_df[self.index_df.split == split]

        self.transform = transform
        self.n_per_class = n_per_class
        self.pairing_strategy = pairing_strategy
        self.include_inverse = include_inverse

        if classes is None:
            self.classes = sorted(self.index_df.class_name.unique())
        else:
            self.classes = classes

        # calculate labels from classes
        self.index_df["label"] = self.index_df.class_name.map(lambda c: self.classes.index(c))

        if pairing_strategy == "product": # cross product of A and B sides
            self.image_paths, self.labels = self.AB_product_image_paths_labels()
        elif pairing_strategy == "session":
            self.image_paths, self.labels = self.AB_session_image_paths_labels()
        else:
            print(f"Unrecognised pairing strategy: {pairing_strategy}")

    def AB_session_image_paths_labels(self):
        image_paths = []
        labels = []
        for label, group in self.index_df.groupby("label"):
            class_image_paths = []
            for session, pair in group.groupby("session"):
                #print(session)
                #print(pair)
                if(len(pair) != 2): # error
                    print(f"ERROR: session {session} length is {len(pair)} for A and B pairing")
                    continue # skip
                class_image_paths.append((pair[pair.fabric_side == "A"].iloc[0].image_path,
                                          pair[pair.fabric_side == "B"].iloc[0].image_path))

            # limit to a random n images per class
            if self.n_per_class is not None:
                class_image_paths = random.sample(class_image_paths, min(len(class_image_paths), self.n_per_class))

            # add to the dataset
            image_paths.extend(class_image_paths)
            labels.extend([label] * len(class_image_paths))
        return image_paths, labels

    def AB_product_image_paths_labels(self):
        image_paths = []
        labels = []

        # per group
        for label, group in self.index_df.groupby("label"):
            A_df = group[group.fabric_side == "A"]
            B_df = group[group.fabric_side == "B"]

            # combine A,B ...
            class_image_paths = list(itertools.product(A_df.image_path, B_df.image_path))
            if self.include_inverse: # ... and B,A pairs (if include_inverse)
                class_image_paths.extend(list(itertools.product(B_df.image_path, A_df.image_path)))
            #print(class_image_paths)

            if self.n_per_class is not None: # limit to a random n images per class
                class_image_paths = random.sample(class_image_paths, min(len(class_image_paths), self.n_per_class))

            # add to the dataset
            image_paths.extend(class_image_paths)
            labels.extend([label] * len(class_image_paths))
            # print(label, len(group), len(class_image_paths), len(image_paths), len(labels))
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image1 = Image.open(self.image_paths[idx][0]).convert("RGB")
        #name1 = pathlib.Path(self.image_paths[idx][0]).name
        name1 = self.image_paths[idx][0]
        image2 = Image.open(self.image_paths[idx][1]).convert("RGB")
        #name2 = pathlib.Path(self.image_paths[idx][1]).name
        name2 = self.image_paths[idx][1]

        label = self.labels[idx]

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, label, name1, name2

index_paths = {
    'sml_lab': 'data/training/SML/sml_lab.csv',
    'sml_lab_test': 'data/training/SML/sml_lab_test_eval.csv',
    'sml_expo_eval': 'data/evaluation/SML/sml_expo_eval_10-14.csv'
}

def get_dataset(dataset, mode, split, n_per_class, transform):
    if mode == "single": # single Dataset
        return SingleImageDataset(index_paths[dataset], 
                                  split=split, 
                                  n_per_class=n_per_class, 
                                  transform=transform)
    elif mode == "dual":
        return DualImageDataset(index_paths[dataset], 
                                split=split,
                                n_per_class=n_per_class,
                                transform=transform)
