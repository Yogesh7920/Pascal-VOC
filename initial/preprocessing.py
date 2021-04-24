import os

from config import Config


def annot_preprocessing(text):
    text = text.lower()
    captions = text.split('\n')
    captions_unique = list(set(captions))
    text = '\n'.join(captions_unique)
    return text


def annots_preprocessing():
    annots1 = Config.abstract50s['annotations']
    annots2 = Config.pascal50s['annotations']

    for annot in os.listdir(annots1):
        path = os.path.join(annots1, annot)
        with open(path, 'r') as f:
            text = f.read()

        text = annot_preprocessing(text)

        with open(path, 'w') as f:
            f.write(text)

    for annot in os.listdir(annots2):
        path = os.path.join(annots2, annot)
        with open(path, 'r') as f:
            text = f.read()

        text = annot_preprocessing(text)

        with open(path, 'w') as f:
            f.write(text)


if __name__ == '__main__':
    pass
    # annots_preprocessing()
