import os
from matplotlib import pyplot as plt


def show_example(dataset, index):
    image_path = os.path.join(dataset['images'], os.listdir(dataset['images'])[index])
    annot_path = os.path.join(dataset['annotations'], os.listdir(dataset['annotations'])[index])

    plt.imshow(plt.imread(image_path))
    with open(annot_path) as f:
        print(f.read())
