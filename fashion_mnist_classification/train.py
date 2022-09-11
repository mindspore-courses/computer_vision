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

import argparse

from mindspore import context
from mindspore import Model
from mindspore import nn

from mindspore.nn import Accuracy
from mindspore.train.callback import CheckpointConfig, LossMonitor, ModelCheckpoint

from dataset import create_dataset
from lenet import LeNet5


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
