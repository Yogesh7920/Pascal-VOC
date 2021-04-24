import os
import shutil

import numpy as np
import requests
import scipy.io
from matplotlib import pyplot as plt

from config import Config


def organize_dataset():
    abstract50S = scipy.io.loadmat("Datasets/abstract50S.mat")
    pascal50S = scipy.io.loadmat("Datasets/pascal50S.mat")

    def create_dir(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    abstract50S_path = {}
    abstract50S_path["images"] = "Datasets/abstract50S/images/"
    abstract50S_path["annotations"] = "Datasets/abstract50S/annotations/"
    create_dir(abstract50S_path["images"])
    create_dir(abstract50S_path["annotations"])

    pascal50S_path = {}
    pascal50S_path["images"] = "Datasets/pascal50S/images/"
    pascal50S_path["annotations"] = "Datasets/pascal50S/annotations/"
    create_dir(pascal50S_path["images"])
    create_dir(pascal50S_path["annotations"])

    pascal50S = pascal50S['train_sent_final'].flatten()
    abstract50S = abstract50S['abs_sent'].flatten()

    ds = []
    for (desc, img) in abstract50S:
        ds.append((img, desc))

    pascal50S = list(pascal50S)
    abstract50S = ds

    def get_dataset(dataset, dataset_path):
        for (i, (img, desc)) in enumerate(dataset, start=1):
            desc = desc.flatten()
            url = img[0]
            response = requests.get(url, stream=True)
            img_save_path = dataset_path["images"] + '{:04d}.jpg'.format(i)
            desc_save_path = dataset_path["annotations"] + '{:04d}.npy'.format(i)
            with open(img_save_path, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            del response
            with open(desc_save_path, 'wb') as out_file:
                np.save(out_file, desc)

    get_dataset(pascal50S, pascal50S_path)
    get_dataset(abstract50S, abstract50S_path)


def show_example(dataset, index):
    image_path = os.path.join(dataset['images'], os.listdir(dataset['images'])[index])
    annot_path = os.path.join(dataset['annotations'], os.listdir(dataset['annotations'])[index])

    plt.imshow(plt.imread(image_path))
    data = np.load(annot_path, allow_pickle=True)
    for row in data:
        print(row[0])
