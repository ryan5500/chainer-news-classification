# -*- coding: utf-8 -*-
from sklearn.cross_validation import train_test_split
import corpus
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset


# Network definition
class MLP(chainer.Chain):

    def __init__(self, in_units, n_units, n_label):
        super(MLP, self).__init__(
            l1=L.Linear(in_units, n_units),  # first layer
            l2=L.Linear(n_units, n_units),  # second layer
            l3=L.Linear(n_units, n_label),  # output layer
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def main():
    # 辞書の読み込み
    dictionary = corpus.get_dictionary(create_flg=False)
    # 記事の読み込み
    contents = corpus.get_contents()

    # 特徴抽出
    data_train = []
    label_train = []
    for file_name, content in contents.items():
        data_train.append(corpus.get_vector(dictionary, content))
        label_train.append(corpus.get_class_id(file_name))

    data_train_s, data_test_s, label_train_s, label_test_s = train_test_split(data_train, label_train, test_size=0.5)

    N_test = len(data_test_s)         # test data size
    N = len(data_train_s)             # train data size
    in_units = len(data_train_s[0])  # 入力層のユニット数 (語彙数)

    n_units = 1000 # 隠れ層のユニット数
    n_label = 9    # 出力層のユニット数

    #モデルの定義
    model = L.Classifier(MLP(in_units, n_units, n_label))

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    np_data_train_s = np.array(data_train_s, dtype=np.float32)
    np_label_train_s = np.array(label_train_s, dtype=np.int32)
    np_data_test_s = np.array(data_test_s, dtype=np.float32)
    np_label_test_s = np.array(label_test_s, dtype=np.int32)

    train_iter = chainer.iterators.SerialIterator(tuple_dataset.TupleDataset(np_data_train_s, np_label_train_s), 100)
    test_iter = chainer.iterators.SerialIterator(tuple_dataset.TupleDataset(np_data_test_s, np_label_test_s), 100, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=-1)
    trainer = training.Trainer(updater, (20, 'epoch'), out='result')

    trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot())
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()


if __name__ == '__main__':
    main()
