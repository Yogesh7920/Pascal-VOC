import os
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
import spacy

from config import Config

nlp = spacy.load('en_core_web_sm')


def get_image(dataset, index):
    image_path = dataset['images'][index]
    return plt.imread(image_path)


def get_annots(dataset, index):
    annot_path = dataset['annotations'][index]
    with open(annot_path) as f:
        text = f.read()

    return text


def show_example(dataset, index):
    annot = get_annots(dataset, index)
    print(annot)
    plt.imshow(get_image(dataset, index))


def find_indices(lst, item):
    return [i for i, x in enumerate(lst) if x == item]


def processing(sentence, target):
    people = {'person', 'man', 'woman', 'boy', 'girl', 'lady', 'kid', 'human',
              'child', 'adult', 'guy', 'people'}
    birds = {'bird', 'parrot', 'peacock', 'duck', 'chicken', 'hen', 'sparrow',
             'pigeon', 'crow', 'eagle', 'kingfisher', 'turkey', 'ostrich',
             'cuckoo', 'woodpecker', 'vulture'}

    doc = nlp(sentence)
    tokens = [token.lemma_.lower() for token in doc]
    raw_tokens = [token.text.lower() for token in doc]
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
        elif label == 'bird':
            if birds.intersection(tokens) != set():
                target[Config.label_dict[label]] = 1
        elif label == 'dog':
            if 'dog' in tokens:
                inds = find_indices(tokens, 'dog')
                inds = list(map(lambda x: x-1, inds))
                flag = True
                for i in inds:
                    if tokens[i] == 'hot':
                        flag = False
                if flag:
                    target[Config.label_dict[label]] = 1

        elif label == 'person':
            if people.intersection(tokens) != set():
                ind = find_indices(tokens, 'person')
                if len(ind) and raw_tokens[ind[0]] == 'persons':
                    continue
                target[Config.label_dict[label]] = 1

        elif label in tokens:
            target[Config.label_dict[label]] = 1

    for token in doc.ents:
        if token.label_ == 'PERSON':
            target[Config.label_dict['person']] = 1


def get_targets(dataset):
    targets = []
    for i in range(dataset['size']):
        annots = get_annots(dataset, i).split('\n')
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
    return {k for k, v in d.items() if v == 1}


def get_images(dataset):
    images = []
    for i in range(dataset['size']):
        image = get_image(dataset, i)
        images.append(image)

    images = np.array(images)
    return images
