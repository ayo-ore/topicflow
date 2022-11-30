import numpy as np
import os
import pickle
import sys
import tensorflow as tf

from abc import abstractmethod
from glob import glob
from scipy.stats import wasserstein_distance
from subprocess import call
from tensorflow.keras import callbacks


# vectorized wasserstein distance function
np_wasserstein = np.vectorize(wasserstein_distance, signature='(i),(j)->()')


def check_fractions(*args):
    assert sum(args) == 1, "The data splits must sum to one."


def check_purities(purities):
    assert all((0 <= p <= 1 for p in purities)), \
        "Purities must be in the range [0,1]."
    assert purities[0] != purities[1], "The mixture purities must be different"


def create_tag(purities, train_frac):
    """Creates a label for the training settings."""
    purity_tag = 'P' + '_'.join(map(lambda p: f"{p*100:n}", purities))
    return purity_tag + f"_T{train_frac*100:n}"


def load_topic_flows(path, arch):
    """
    Loads quark/gluon flows from the specified training output directory.
    """
    model_config = pickle.load(open(
        os.path.join(path, 'model_config.p'), 'rb'
    ))
    qmodel = arch(**model_config, training=False)
    gmodel = arch(**model_config, training=False)
    qmodel.load_weights(
        os.path.join(path, 'qweights')
    ).expect_partial().assert_existing_objects_matched()
    gmodel.load_weights(
        os.path.join(path, 'gweights')
    ).expect_partial().assert_existing_objects_matched()
    return qmodel, gmodel


class VariableWarmUp(callbacks.Callback):
    """
    A callback that 'warms up' a model parameter over the specified number
    of epochs by linearly interpolating from a value of 0.
    """

    def __init__(self, param_name, warmup_epochs):
        self.param_name = param_name
        self.warmup_epochs = warmup_epochs

    def on_train_begin(self, logs=None):
        self.target = tf.keras.backend.get_value(
            getattr(self.model, self.param_name)
        )

    def on_epoch_begin(self, epoch, logs=None):

        val = (
            self.get_val(epoch) if epoch < self.warmup_epochs
            else self.target
        )
        if epoch <= self.warmup_epochs:
            tf.keras.backend.set_value(
                getattr(self.model, self.param_name), val
            )

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


def submit_train_job(model, tag, run, args):
    """
    Submits a job to the batch system that runs the parent script with the
    same configuration.
    """

    if args.nopca:
        model += '_nopca'
    model_dir = os.path.join(args.savedir, f'd{args.dimension}', model, tag)

    # determine run number
    prev_runs = glob(os.path.join(model_dir, 'run*'))
    next_run = max([int(p.split('run_')[-1]) for p in prev_runs]) + 1 \
        if prev_runs else 0
    setup_cmd = 'ml fosscuda/2020b python/3.9.6 cudnn/8.2.1.32-cuda-11.3.1'
    drop_args = ['-q', '--queue', args.queue, '-s', '--savedir', args.savedir]
    rundir = os.path.join(
        model_dir, f'run_{next_run + (run if args.dry else 0)}'
    )
    script_cmd = 'python ' + ' '.join(
        [a for a in sys.argv if a not in drop_args]
    ) + f" --savedir {rundir}"

    log_dir = os.path.join(rundir, 'log')
    if not args.dry:
        os.makedirs(log_dir, exist_ok=False)

    qos = ' -q gpgpuresplat' if args.queue == 'gpgpu' else ''
    cmd = (
        f"sbatch -p {args.queue}{qos} --mem {args.memory} -N 1 -c 4 --gpus 1 "
        f"-t {args.time} -J {model}_{tag} "
        f"-e {os.path.join(log_dir, '%x')}.err "
        f"-o {os.path.join(log_dir, '%x')}.out "
        f"--wrap \"{setup_cmd}; {script_cmd}\""
    )
    print(f'[CMD] {cmd}')
    if not args.dry:
        call(cmd, shell=True, executable='/bin/bash')