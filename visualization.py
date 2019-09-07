import keras.backend as K
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import numpy as np


class TensorBoardLogger(object):
    def __init__(self, log_dir):
        if K.backend() != 'tensorflow':
            raise RuntimeError('TensorBoardLogger only works '
                               'with the TensorFlow backend.')
        self.log_dir = log_dir
        self.compiled = False
        self.merged = None
        self.writer = None

    @staticmethod
    def add_log_scalar(name, scalar):
        tf.summary.scalar(name, scalar)

    def compile(self, model=None):
        assert not self.compiled
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_dir)
        if model is not None:
            self.augment_model(model)
        self.compiled = True

    def augment_model(self, model):
        model._make_train_function()
        model.train_function.inputs.append(model.inputs + [K.learning_phase()])
        model.train_function.outputs.append(self.merged)

    def write_log(self, name, value, global_step):
        assert self.compiled
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        self.writer.add_summary(summary, global_step)
        self.writer.flush()

    def write_summary(self, summary_value, global_step):
        assert self.compiled
        self.writer.add_summary(summary_value, global_step=global_step)
        self.writer.flush()


from keras.optimizers import SGD


def main():
    model = Sequential()
    model.add(Dense(output_dim=3, input_dim=2, activation='tanh'))
    model.add(Dense(output_dim=10, activation='tanh'))
    model.add(Dense(output_dim=4))

    model.compile(optimizer=SGD(1.0), loss='mse')

    # tf.summary.scalar('output', K.mean(model.output))
    tf.summary.scalar('loss_xyz', model.model.total_loss)
    merged = tf.summary.merge_all()

    log_dir = '/tmp/tensorboard-test-b/'
    writer = tf.summary.FileWriter(log_dir)

    model.model._make_train_function()

    model.model.train_function.inputs.append(model.model.inputs + [K.learning_phase()])
    model.model.train_function.outputs.append(merged)

    x = np.random.uniform(-1, 1, (10, 2))
    y = np.random.uniform(-1, 1, (10, 4))
    for i in range(0, 1000):
        result, summary_value = model.train_on_batch(x=x,
                                                     y=y)

        if i%10 == 0:
            writer.add_summary(summary_value, global_step=i)
            writer.flush()

        print('result:', result)


if __name__ == '__main__':
    main()

    import gc

    gc.collect()
