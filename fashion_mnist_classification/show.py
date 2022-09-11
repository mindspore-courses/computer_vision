#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------
# @Version : 1.0
# @Author : xingchaolong
# @For : 
# -------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import mindspore.dataset as ds


if __name__ == "__main__":
    fashion_mnist_dataset_dir = "./data/"
    fashion_mnist_dataset = ds.FashionMnistDataset(dataset_dir=fashion_mnist_dataset_dir, num_samples=3)
    fashion_mnist_it = fashion_mnist_dataset.create_dict_iterator()
    data = next(fashion_mnist_it)

    plt.imshow(data['image'].asnumpy().reshape(28, 28), cmap='gray')
    plt.title(data['label'].asnumpy(), fontsize=20)
    plt.show()
