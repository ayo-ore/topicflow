import numpy as np
import os
import pickle
import tensorflow as tf

from abc import abstractmethod
from scipy.stats import wasserstein_distance
from tensorflow.keras import callbacks, optimizers

def check_fractions(*args):
    assert sum(args) == 1, "The data splits must sum to one."

def check_purities(purities):
    assert all((0 <= p <= 1 for p in purities)), \
        "Purities must be in the range [0,1]."
    assert purities[0] != purities[1], "The mixture purities must be different"

def create_tag(purities, train_frac):
    purity_tag = 'P' + '_'.join(map(lambda p: f"{p*100:n}", purities))
    return purity_tag + f"_T{train_frac*100:n}"

np_wasserstein = np.vectorize(wasserstein_distance, signature='(i),(j)->()')

def load_topic_flows(path, arch):
    model_config = pickle.load(open(
        os.path.join(path, 'model_config.p'), 'rb'
    ))
    model1 = arch(**model_config, training=False)
    model2 = arch(**model_config, training=False)
    model1.load_weights(
        os.path.join(path, 'weights1')
    ).expect_partial().assert_existing_objects_matched()
    model2.load_weights(
        os.path.join(path, 'weights2')
    ).expect_partial().assert_existing_objects_matched()
    return model1, model2
         

def demix_topics(mix1, mix2, num_bins, poly_degree):

    # determine bins by quantiles
    total = np.hstack([mix1, mix2])
    quantiles = np.linspace(0, 1, num_bins+1)
    bin_edges = np.quantile(total, quantiles)

    # use bins to histogram mixture
    vals1, _ = np.histogram(mix1, bins=bin_edges, density=True)
    vals2, _ = np.histogram(mix2, bins=bin_edges, density=True)

    # calculate histogram ratio and error
    ratio = vals1 / vals2

    # polynomial fit
    quantiles_centers = (quantiles[1:] + quantiles[:-1])/2
    polyfit = np.polynomial.Legendre.fit(
        quantiles_centers, ratio, poly_degree, domain=[0, 1]
    )

    # determine reducibility factors by min/max val
    x, y = polyfit.linspace(500)
    k_qg, k_gq = min(y), 1/max(y)

    # calculate corresponging fractions
    fq = (1 - k_qg) / (1 - k_qg * k_gq)
    fg = k_gq * fq

    return {'fractions': (fq, fg), 'reducibilities': (k_qg, k_gq)}


class OneCycleSchedule(optimizers.schedules.LearningRateSchedule):

    def __init__(self, max_lr, epochs, init_frac=0.1, final_frac=0.001, period=0.8):

        self.max_lr = max_lr
        self.init_lr = self.max_lr * init_frac
        self.final_lr = self.max_lr * final_frac
        self.epochs = epochs
        self.period = period

    @tf.function
    def __call__(self, step):
        frac = step/self.epochs
        if frac < self.period/2:
            return self.init_lr + (2*frac/self.period)*(self.max_lr - self.init_lr)
        elif frac < self.period:
            return self.max_lr + (2*frac/self.period-1)*(self.init_lr - self.max_lr)
        else:
            return self.init_lr + (frac-self.period)/(1-self.period)*(self.final_lr - self.init_lr)


class VariableWarmUp(callbacks.Callback):
    """
    A callback that 'warms up' a model parameter over the specified number
    of epochs by linearly interpolating from a value of 0.
    """

    def __init__(self, param_name, warmup_epochs):
        self.param_name = param_name
        self.warmup_epochs = warmup_epochs

    def on_train_begin(self, logs=None):
        self.target = tf.keras.backend.get_value(getattr(self.model, self.param_name))

    def on_epoch_begin(self, epoch, logs=None):

        val = self.get_val(epoch) if epoch < self.warmup_epochs else self.target
        if epoch <= self.warmup_epochs:
            tf.keras.backend.set_value(getattr(self.model, self.param_name), val)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs[self.param_name] = getattr(self.model, self.param_name)

    @abstractmethod
    def get_val(self, epoch, logs=None):
        pass


class LinearWarmUp(VariableWarmUp):

    def get_val(self, epoch, logs=None):
        return self.target * float(epoch) / self.warmup_epochs


class PowerWarmUp(VariableWarmUp):

    def get_val(self, epoch, logs=None):
        return self.target / (1 + (self.warmup_epochs - epoch))


class ExponentialWarmUp(VariableWarmUp):

    def get_val(self, epoch, logs=None):
        return self.target * np.exp(epoch - self.warmup_epochs)


class ZeroWarmUp(VariableWarmUp):

    def get_val(self, epoch, logs=None):
        return 0.
