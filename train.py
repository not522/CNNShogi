# -*- coding: utf-8 -*-

import argparse

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions

import net
from shogi import get_data


class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret


def main():
    parser = argparse.ArgumentParser(description='CNN Shogi:')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of data in each mini-batch')
    parser.add_argument('--alpha', '-a', type=float, default=0.001,
                        help='Alpha parameter of Adam')
    parser.add_argument('--epoch', '-e', type=int, default=10,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--datasize', '-d', type=int, default=1000,
                        help='Number of data')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    train, test = get_data(args.datasize)
    model = net.Model()
    classifier = L.Classifier(model)

    # GPUを使う場合
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        classifier.to_gpu()

    # trainerの設定
    optimizer = chainer.optimizers.Adam(alpha=args.alpha)
    optimizer.setup(classifier)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(TestModeEvaluator(test_iter, classifier, device=args.gpu))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot_object(
        target=model, filename='snapshot', trigger=(args.epoch, 'epoch')))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
