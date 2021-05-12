from config import Config


def annot_preprocessing(text):
    captions = text.split('\n')
    captions_unique = list(set(captions))
    text = '\n'.join(captions_unique)
    return text


def annots_preprocessing():
    annots1 = Config.abstract50s['annotations']
    annots2 = Config.pascal50s['annotations']

    for dataset_annot in [annots1, annots2]:
        for annot in dataset_annot:

            with open(annot, 'r') as f:
                text = f.read()

            text = annot_preprocessing(text)

            with open(annot, 'w') as f:
                f.write(text)


if __name__ == '__main__':
    pass
    # annots_preprocessing()
