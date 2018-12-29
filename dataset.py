import torch.utils.data
import numpy as np
from mnist import MNIST


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        mnist = MNIST(path, return_type='numpy', gz=True)
        images, labels = mnist.load_training()
        images = (images / 255 * 2 - 1).astype(np.float32)
        images = images.reshape((images.shape[0], 28, 28))
        labels = labels.astype(np.int64)

        self.images = images
        self.labels = labels

    def __getitem__(self, item):
        image = self.images[item]
        label = self.labels[item]

        image = image.reshape((1, *image.shape))

        return image, label

    def __len__(self):
        return len(self.images)
