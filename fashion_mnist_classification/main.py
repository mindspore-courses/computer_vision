#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------
# @Version : 1.0
# @Author : xingchaolong
# @For : MindSpore FashionMnist LeNet Example.
# -------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import mindspore.dataset as ds
import mindspore.nn as nn
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV

from mindspore import context
from mindspore import dtype as mstype
from mindspore import Model
from mindspore.common.initializer import Normal
from mindspore.dataset.vision import Inter
from mindspore.nn import Accuracy
from mindspore.train.callback import CheckpointConfig, LossMonitor, ModelCheckpoint


def create_dataset(data_path, usage="train", batch_size=32, repeat_size=1, num_parallel_workers=1):
    # 定义数据集
    fashion_mnist_ds = ds.FashionMnistDataset(data_path, usage=usage)
    resize_height, resize_width = 28, 28
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # 定义所需要操作的map映射
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # 使用map映射函数，将数据操作应用到数据集
    fashion_mnist_ds = fashion_mnist_ds.map(
        operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    fashion_mnist_ds = fashion_mnist_ds.map(
        operations=[resize_op, rescale_op, rescale_nml_op, hwc2chw_op],
        input_columns="image", num_parallel_workers=num_parallel_workers)

    # 进行shuffle、batch、repeat操作
    buffer_size = 10000
    fashion_mnist_ds = fashion_mnist_ds.shuffle(buffer_size=buffer_size)
    fashion_mnist_ds = fashion_mnist_ds.batch(batch_size, drop_remainder=True)
    fashion_mnist_ds = fashion_mnist_ds.repeat(count=repeat_size)

    return fashion_mnist_ds


class LeNet5(nn.Cell):
    """
    Lenet网络结构
    """
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        # 定义所需要的运算
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 4 * 4, 256, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(256, 128, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(128, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        # 使用定义好的运算构建前向网络
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def train_net(model, epoch_size, data_path, batch_size, repeat_size, ckpt_cb, sink_mode):
    """定义训练的方法"""
    # 加载训练数据集
    ds_train = create_dataset(data_path, usage="train", batch_size=batch_size, repeat_size=repeat_size)
    model.train(epoch_size, ds_train, callbacks=[ckpt_cb, LossMonitor(125)], dataset_sink_mode=sink_mode)


def test_net(model, data_path):
    """定义验证的方法"""
    ds_eval = create_dataset(data_path, usage="test")
    acc = model.eval(ds_eval, dataset_sink_mode=False)
    print("acc: {}".format(acc), flush=True)


def run(data_path, device_target="CPU", batch_size=32, train_epoch=5, dataset_size=1):
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)

    net = LeNet5()
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)

    # 设置模型保存参数
    config_ck = CheckpointConfig(save_checkpoint_steps=100, keep_checkpoint_max=10)
    # 应用模型保存参数
    ckpt_cb = ModelCheckpoint(prefix="lenet_ckpt", config=config_ck)

    model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    train_net(model, train_epoch, data_path, batch_size, dataset_size, ckpt_cb, False)
    test_net(model, data_path)


def main():
    parser = argparse.ArgumentParser(description='MindSpore FashionMnist LeNet Example.')
    parser.add_argument("--data_path", type=str, required=True, help="fashion mnist data path.")
    parser.add_argument("--device_target", type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'],
                        help="target device")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size.")
    parser.add_argument("--train_epoch", type=int, default=5, help="train epoch.")
    parser.add_argument("--dataset_size", type=int, default=1, help="dataset size.")

    args = parser.parse_args()

    run(
        data_path=args.data_path,
        device_target=args.device_target,
        batch_size=args.batch_size,
        train_epoch=args.train_epoch,
        dataset_size=args.dataset_size
    )


if __name__ == "__main__":
    main()
