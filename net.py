# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L


class Model(chainer.Chain):

    # モデルの設定
    def __init__(self):
        convs = chainer.ChainList(
            *[L.Convolution2D(None, 16, 5, pad=2) for i in range(4)])
        bns = chainer.ChainList(
            *[L.BatchNormalization(16) for i in range(4)])
        super(Model, self).__init__(
            convs=convs,
            bns=bns,
            conv=L.Convolution2D(None, 27, 5, pad=2),
        )
        self.train = True

    # モデルを呼び出す
    def __call__(self, x):
        for i in range(4):
            x = self.convs[i](x)
            x = self.bns[i](x, test=not self.train)
            x = F.relu(x)
        x = self.conv(x)
        x = F.reshape(x, (x.data.shape[0], -1))
        return x
