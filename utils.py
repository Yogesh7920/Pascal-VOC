import os
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
import spacy

from config import Config

nlp = spacy.load('en_core_web_sm')


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


def processing(sentence, target):
    doc = nlp(sentence)
    tokens = {token.text.lower() for token in doc}
    for label in Config.labels:
        if label == 'dining table':
            if 'table' in tokens:
                target[Config.label_dict[label]] = 1
        elif label == 'potted plant':
            if 'plant' in tokens:
                target[Config.label_dict[label]] = 1
        elif label == 'aeroplane':
            if any('plane' in string for string in tokens):
                target[Config.label_dict[label]] = 1

        elif label == 'motorbike':
            if any('bike' in string for string in tokens):
                target[Config.label_dict[label]] = 1

        elif label in tokens:
            target[Config.label_dict[label]] = 1

    for token in doc.ents:
        if token.label_ == 'PERSON':
            target[Config.label_dict['person']] = 1


def get_targets():
    targets = []
    for i in range(Config.abstract50s['size']):
        annots = get_annots(Config.abstract50s, i).split('\n')
        target = [0 for _ in range(len(Config.labels))]
        for annot in annots:
            processing(annot, target)

        targets.append(target)

    targets = np.array(targets)
    expanded_targets = deepcopy(targets)
    # Tv/Monitor are the same class.
    targets[:, -2] = np.bitwise_or(targets[:, -1], targets[:, -2])
    targets = targets[:, :-1]

    return expanded_targets, targets


def check_target(dataset, targets, index):
    plt.imshow(get_image(dataset, index))
    d = dict(zip(Config.labels, targets[index]))
    return {k for k, v in d.items() if v==1}


def get_images(dataset):
    images = []
    for i in range(dataset['size']):
        image = get_image(Config.abstract50s, i)
        images.append(image)

    images = np.array(images)
    return images
