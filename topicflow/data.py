import numpy as np
import os
from glob import glob
from scipy.optimize import linprog
from tensorflow.data import Dataset, AUTOTUNE
from topicflow import utils

EFP_DIR = '~/.topic_flow/efps'

def get_file_num(path):
    """Retrieves the file number of the given `.npz` file."""
    base = path.split('.npz')[0]
    numstr = base.split('_')[-1]
    return int(numstr)

def get_file_splits(directory, fractions):
    """
    Returns a dictionary mapping data splits to lists of `.npz` files from the
    given directory. Splits are inherited from the keys of `fractions`.
    """
    assert sum(fractions.values()) == 1, \
        "The fractions of all splits must sum to one."

    # find and count all nprecords
    nprecords = glob(os.path.join(directory, '*.npz'))
    num_files = len(nprecords)
    assert num_files > 0, f"Could not find any numpy records in {directory}"
    assert all([
        n == int(n)
        for n in map(lambda f: f * num_files / 2, fractions.values())
    ]), (
        f"Not possible to split as requested with {num_files} files, minimum "
        f"unit is {2/num_files:n}. Consider sharding data for smaller splits."
    )

    # get number of files in each split
    edges, consumed = {}, 0
    for split, fraction in fractions.items():
        start = consumed * num_files / 2
        consumed += fraction
        end = int(consumed * num_files / 2)
        edges[split] = (start, end)

    # split files
    file_splits = {
        split: list(
            filter(lambda p: start <= get_file_num(p) < end, nprecords)
        ) for split, (start, end) in edges.items()
    }
    return file_splits
    
def preprocess(quarks, gluons, pca, norms):

    # remove rows with any zeros for log scaling
    quarks = quarks[(quarks.min(axis=1) != 0)]
    gluons = gluons[(gluons.min(axis=1) != 0)]
    
    # stack quarks and gluons
    stack = np.vstack([quarks, gluons])
    np.log(stack, out=stack)
    
    # shift to zero mean
    mean = norms['mean'] or stack.mean(axis=0)
    np.subtract(stack, mean, out=stack)

    if pca:
        # transform to principal component basis
        evecs = norms['evecs'] or np.linalg.eig(np.cov(stack.T))[1]
        np.dot(stack, evecs, out=stack)
    
    # scale to unit standard deviation
    std = norms['std'] or stack.std(axis=0)
    np.divide(stack, std, out=stack)
    
    # convert to float32
    stack = stack.astype(np.float32, copy=False)
    
    norms = {'mean': mean, 'std': std}
    if pca:
        norms['evecs'] = evecs
    
    return np.split(stack, [len(quarks)]), norms

def get_component_cutoffs(purities):
    
    num_mixtures = len(purities)
    if num_mixtures == 1:
        quark_cutoffs = np.array(purities)
        return quark_cutoffs, 1 - quark_cutoffs
        
    num_vars = 2 * num_mixtures

    # ub1: cannot sample beyond balanced quota
    A_ub1 = np.hstack([np.eye(num_mixtures)]*2)
    b_ub1 = 2*np.ones(num_mixtures)/num_mixtures
    
    # ub2: cannot sample beyond available components
    first_row = np.hstack([np.ones(num_mixtures), np.zeros(num_mixtures)])
    second_row = 1 - first_row
    A_ub2 = np.vstack([first_row, second_row])
    b_ub2 = np.ones(2)
    
    # linear program
    diag_purities = np.diag(purities)
    st = linprog(
        c = -np.ones(num_vars),           # maximize assigned examples
        A_ub = np.vstack([A_ub1, A_ub2]), # cannot sample beyond balanced quota
        b_ub = np.hstack([b_ub1, b_ub2]),
        A_eq = np.hstack([                # assign correct purities
            diag_purities-np.eye(len(purities)),
            diag_purities
        ]), 
        b_eq = np.zeros(num_mixtures),
        bounds = [(0,1)] * num_vars
    )
    
    quark_sizes, gluon_sizes = np.split(st.x, 2)
    if sum(purities) % 1 != 0:
        print(
            f"WARNING: Purities are unbalanced. The mixtures have different"
            f" sizes: {quark_sizes + gluon_sizes}."
        )
    quark_cutoffs = np.cumsum(quark_sizes)
    gluon_cutoffs = np.cumsum(gluon_sizes)
    
    return quark_cutoffs, gluon_cutoffs


def get_samples_from_disk(directory, fractions, purities, pca, only=None):

    # check that we have valid purities
    utils.check_purities(purities)

    # determine component cutoffs
    quark_cutoffs, gluon_cutoffs = get_component_cutoffs(purities)

    # load preprocessing arrays
    arrays = {}
    file_splits = get_file_splits(directory, fractions)
    norms = {k: None for k in ('mean', 'std', 'evecs')}
    for split, files in file_splits.items():
        if split == only or not only:

            # load current split
            quarks = np.vstack(
                [np.load(f)['data'] for f in files if 'quark' in f]
            )
            gluons = np.vstack(
                [np.load(f)['data'] for f in files if 'gluon' in f]
            )

            # preprocess
            (quarks, gluons), norms = preprocess(
                quarks, gluons, pca=pca, norms=norms
            )

            # scale mixture boundaries to current split
            quark_cutoff_idcs = (quark_cutoffs * len(quarks)).astype(int)
            gluon_cutoff_idcs = (gluon_cutoffs * len(gluons)).astype(int)

            # create mixture arrays
            quark_components = np.split(quarks, quark_cutoff_idcs)[:-1]
            gluon_components = np.split(gluons, gluon_cutoff_idcs)[:-1]

            mixtures = [
                np.vstack([qarr, garr])
                for qarr, garr in zip(quark_components, gluon_components)
            ]

            arrays[split] = mixtures

    return arrays


def get_mixture_datasets(
        dim,
        purities,
        fractions,
        labels=True,
        pca=True,
        batch_size=500,
        shuffle_buffer=5e4,
        efp_dir=EFP_DIR
        ):
    """
    Returns a dictionary mapping data splits to `tf.data.Dataset`s. Splits are
    inherited from the keys of `fractions`. The dataset composition is
    determined by the `purities` argument -- a list of quark purities for each
    mixture.
    """

    # ensure provided purities are compatible with the labelling
    assert len(purities) == 2 if labels else 1

    # load quark/gluon samples
    arrays = get_samples_from_disk(
        os.path.join(efp_dir, f'd{dim}'), fractions, purities, pca=pca
    )

    # create function to shuffle and batch datasets
    def shuffle_and_batch(dataset: Dataset) -> Dataset:
        dataset = dataset.shuffle(
            int(shuffle_buffer), reshuffle_each_iteration=True
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset

    datasets = {}
    rng = np.random.default_rng()
    for split, mixtures in arrays.items():

        # concatenate mixtures
        X = np.vstack(mixtures)
        idcs = np.arange(X.shape[0])
        rng.shuffle(idcs)
        X = Dataset.from_tensor_slices(X[idcs])

        if not labels:
            datasets[split] = X.apply(shuffle_and_batch)
        else:
            y = np.zeros((len(idcs), 2), np.float32)
            y[:mixtures[0].shape[0], 1] = 1
            y[mixtures[0].shape[0]:, 0] = 1
            y = Dataset.from_tensor_slices(y[idcs])

            datasets[split] = Dataset.zip((X, y)).apply(shuffle_and_batch)

    return datasets
