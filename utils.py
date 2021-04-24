import os
from matplotlib import pyplot as plt


def get_image(dataset, index):
    image_path = os.path.join(dataset['images'], os.listdir(dataset['images'])[index])
    return plt.imread(image_path)


def get_annots(dataset, index):
    annot_path = os.path.join(dataset['annotations'], os.listdir(dataset['annotations'])[index])
    with open(annot_path) as f:
        text = f.read()

    return text


def show_example(dataset, index):
    annot = get_annots(dataset, index)
    print(annot)
    plt.imshow(get_image(dataset, index))

