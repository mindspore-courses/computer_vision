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

import numpy as np

from mindspore import load_checkpoint, load_param_into_net

from mindspore import Model
from mindspore import Tensor

from dataset import create_dataset
from lenet import LeNet5


def predict():
    net = LeNet5()

    # 加载已经保存的用于测试的模型
    param_dict = load_checkpoint("lenet_ckpt-5_1875.ckpt")
    # 加载参数到网络中
    load_param_into_net(net, param_dict)

    # 定义测试数据集，batch_size设置为1，则取出一张图片
    fashion_minst_path = "./data"
    ds_test = create_dataset(data_path=fashion_minst_path, usage="test", batch_size=1).create_dict_iterator()
    for i in range(1000):
        data = next(ds_test)

    data = next(ds_test)
    # images为测试图片，labels为测试图片的实际分类
    images = data["image"].asnumpy()
    labels = data["label"].asnumpy()

    model = Model(net)
    # 使用函数model.predict预测image对应分类
    output = model.predict(Tensor(data['image']))
    predicted = np.argmax(output.asnumpy(), axis=1)

    # 输出预测分类与实际分类
    print(f'Predicted: "{predicted[0]}", Actual: "{labels[0]}"')


if __name__ == "__main__":
    predict()
