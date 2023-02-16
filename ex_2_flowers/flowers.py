import numpy
from matplotlib import pyplot

from ex_2_flowers.dataset_initializer import DatasetInitializer

dataset = DatasetInitializer().load_dataset()
for image, label in dataset.take(1):
    pyplot.imshow(image)
