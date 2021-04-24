import os


class Config:

    # File structure

    datasets = 'Datasets'
    abstract50s = {
        'root': os.path.join(datasets, 'abstract50s'),
        'images': os.path.join(datasets, 'abstract50s', 'images'),
        'annotations': os.path.join(datasets, 'abstract50s', 'annotations'),
    }

    pascal50s = {
        'root': os.path.join(datasets, 'pascal50s'),
        'images': os.path.join(datasets, 'pascal50s', 'images'),
        'annotations': os.path.join(datasets, 'pascal50s', 'annotations'),
    }

    # Model Config

    batches = 32
    learning_rate = 0.01
    patience = 10
    epochs = 500

