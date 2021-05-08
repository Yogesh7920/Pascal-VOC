import os


def list_full_path(d):
    return [os.path.join(d, f) for f in sorted(os.listdir(d))]


class Config:
    # File structure

    datasets = 'Datasets'
    abstract50s = {
        'root': os.path.join(datasets, 'abstract50S'),
        'images': list_full_path(os.path.join(datasets, 'abstract50s', 'images')),
        'annotations': list_full_path(os.path.join(datasets, 'abstract50s', 'annotations')),
        'size': len(os.listdir(os.path.join(datasets, 'abstract50S', 'images')))
    }

    pascal50s = {
        'root': os.path.join(datasets, 'pascal50S'),
        'images': list_full_path(os.path.join(datasets, 'pascal50S', 'images')),
        'annotations': list_full_path(os.path.join(datasets, 'pascal50S', 'annotations')),
        'size': len(os.listdir(os.path.join(datasets, 'pascal50S', 'images')))
    }

    labels = ['person', 'bird', 'cat', 'cow', 'dog', 'horse',
              'sheep', 'aeroplane', 'bicycle', 'boat', 'bus', 'car',
              'motorbike', 'train', 'bottle', 'chair', 'dining table',
              'potted plant', 'sofa', 'tv', 'monitor']

    label_dict = {'person': 0,
                  'bird': 1,
                  'cat': 2,
                  'cow': 3,
                  'dog': 4,
                  'horse': 5,
                  'sheep': 6,
                  'aeroplane': 7,
                  'bicycle': 8,
                  'boat': 9,
                  'bus': 10,
                  'car': 11,
                  'motorbike': 12,
                  'train': 13,
                  'bottle': 14,
                  'chair': 15,
                  'dining table': 16,
                  'potted plant': 17,
                  'sofa': 18,
                  'tv': 19,
                  'monitor': 20}
