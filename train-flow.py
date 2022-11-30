#!/usr/bin/env python3

import numpy as np
import os
import pickle
import topicflow as tofl

from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score
from tensorflow.config.threading import set_inter_op_parallelism_threads
from tensorflow.keras import callbacks, optimizers, regularizers


def train(args):

    # set threads
    set_inter_op_parallelism_threads(4)

    # log arguments
    print(vars(args))

    # construct datasets
    fractions = {
        'trn': args.trn_frac, 'val': args.val_frac, 'tst': args.tst_frac
    }
    f1, f2 = sorted(args.purities, reverse=True)
    datasets = tofl.get_mixture_datasets(
        purities=[f1, f2], fractions=fractions, pca=not args.nopca,
        batch_size=args.batch_size, efp_degree=args.efp_degree,
        efp_dir=args.efp_dir or tofl.data.EFP_DIR
    )

    # configure flows
    bijector_config = {'fraction_masked': 0.5} if args.arch == 'rqs' else {
        'final_time': 1.0, 'atol': args.node_atol, 'exact_trace': False
    }
    net_config = {
        'num_residual_blocks': args.num_residual_blocks,
        'hidden_dim': args.hidden_dim,
        'num_block_layers': args.num_block_layers,
        'activation': 'tanh' if args.arch == 'node' else 'relu',
        'regularizer': (
            regularizers.l1(args.weight_decay) if args.weight_decay else None
        ),
        'dropout': args.dropout
    }
    if args.arch == 'rqs':
        net_config.update({
            'num_knots': args.rqs_num_knots,
            'boundary': [-args.rqs_boundary, args.rqs_boundary],
            'min_width': args.rqs_min_width,
            'min_slope': args.rqs_min_slope
        })
    dimension = datasets['trn'].element_spec[0].shape[1]
    model_config = {
        'dim': dimension,
        'num_transform_layers': args.num_transform_layers,
        'batchnorm': False,
        'bijector_config': bijector_config,
        'net_config': net_config
    }

    # build flows
    arch = tofl.NeuralODEFlow if args.arch == 'node' else tofl.RQSCouplingFlow
    qflow = tofl.TopicFlow(dimension, (1-f1)/(1-f2), 0, arch, model_config)
    gflow = tofl.TopicFlow(dimension,         f2/f1, 1, arch, model_config)
    qflow.build([None, dimension])
    gflow.build([None, dimension])
    qflow.compile(optimizers.Adam(
        args.learning_rate, clipvalue=5 if args.arch == 'rqs' else None)
    )
    gflow.compile(optimizers.Adam(
        args.learning_rate, clipvalue=5 if args.arch == 'rqs' else None)
    )

    # callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', min_delta=1e-4, patience=args.patience, verbose=1,
        restore_best_weights=True,
    )
    kappa_warmup = tofl.utils.ZeroWarmUp(
        param_name='kappa', warmup_epochs=args.warmup_epochs
    )
    reduce_on_plateau = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=5,
        verbose=1, min_lr=1e-6
    )
    qtensorboard = callbacks.TensorBoard(
        log_dir=os.path.join(args.savedir, 'summary/qtopic'), write_graph=False,
        update_freq='epoch', profile_batch=0
    )
    gtensorboard = callbacks.TensorBoard(
        log_dir=os.path.join(args.savedir, 'summary/gtopic'), write_graph=False,
        update_freq='epoch', profile_batch=0
    )

    # training
    qflow.fit(
        datasets['trn'], validation_data=datasets['val'].unbatch().batch(5000),
        epochs=args.epochs, verbose=2, callbacks=[
            early_stopping, kappa_warmup, reduce_on_plateau, qtensorboard
        ]
    )
    gflow.fit(
        datasets['trn'], validation_data=datasets['val'].unbatch().batch(5000),
        epochs=args.epochs, verbose=2, callbacks=[
            early_stopping, kappa_warmup, reduce_on_plateau, gtensorboard
        ]
    )

    # save models
    qweights_path = os.path.join(args.savedir, 'qweights')
    gweights_path = os.path.join(args.savedir, 'gweights')
    model_config_path = os.path.join(args.savedir, 'model_config.p')
    if not args.dry:
        print(f'Saving weights to {qweights_path}, {gweights_path}')
        qflow.flow.save_weights(qweights_path)
        gflow.flow.save_weights(gweights_path)
        with open(model_config_path, 'wb') as mf:
            print(f'Saving model config to {model_config_path}')
            pickle.dump(model_config, mf)

    # test models
    test_data = [
        np.vstack(list(tofl.get_mixture_datasets(
            purities=[p], batch_size=args.batch_size, pca=not args.nopca,
            efp_degree=args.efp_degree, fractions={
                'trn': fractions['trn'],
                'val': 0.85-fractions['trn'],
                'tst': 0.15
            }, efp_dir=args.efp_dir or tofl.data.EFP_DIR,
        )['tst'])) for p in [1., 0.]
    ]

    # wassersteins
    test_samples = [
        qflow.flow.sample(150_000).numpy(), gflow.flow.sample(150_000).numpy()
    ]
    test_distances = [
        tofl.utils.np_wasserstein(d.T, s.T)
        for d, s in zip(test_data, test_samples)
    ]
    distance_path = os.path.join(args.savedir, 'wassersteins.p')
    print(f'Saving wasserstein distances to {distance_path}.')
    if not args.dry:
        with open(distance_path, 'wb') as f:
            pickle.dump(test_distances, f)

    # generative classification metrics
    preds = np.hstack([qflow(d) - gflow(d) for d in test_data])
    labels = np.zeros(len(preds))
    labels[:len(test_data[0])] = 1
    metrics = {
        'acc': ((preds > 0).astype(int) == labels).mean(),
        'auc': roc_auc_score(labels, preds)
    }
    metrics_path = os.path.join(args.savedir, 'metrics.p')
    print(f'Saving metrics to {metrics_path}.')
    if not args.dry:
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-P', '--purities', type=float, nargs=2, required=True)
    parser.add_argument('--nopca', action='store_true')
    parser.add_argument('-s', '--savedir', default=os.getcwd())
    parser.add_argument('-D', '--efp_degree', type=int, default=4)
    parser.add_argument('--efp_dir', default=None)
    parser.add_argument('--dry', action='store_true')

    parser.add_argument('-b', '--batch_size', type=int, default=1000)
    parser.add_argument('-e', '--epochs', type=int, default=200)
    parser.add_argument('-p', '--patience', type=int, default=10)
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-w', '--warmup_epochs', type=int, default=0)
    parser.add_argument('-T', '--trn_frac', type=float, default=0.5)
    parser.add_argument('-V', '--val_frac', type=float, default=0.1)
    parser.add_argument('-E', '--tst_frac', type=float, default=0.4)

    parser.add_argument('-L', '--num_transform_layers', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--num_residual_blocks', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_block_layers', type=float, default=2)
    parser.add_argument('--arch', choices=('node', 'rqs'), default='node')
    parser.add_argument('--node_atol', type=float, default=1e-6)
    parser.add_argument('--rqs_num_knots', type=int, default=8)
    parser.add_argument('--rqs_boundary', type=float, default=6)
    parser.add_argument('--rqs_min_width', type=float, default=1e-4)
    parser.add_argument('--rqs_min_slope', type=float, default=1e-4)

    parser.add_argument('-q', '--queue', default=None)
    parser.add_argument('-m', '--memory', default='20G')
    parser.add_argument('-t', '--time', default='48:00:00')
    parser.add_argument('-n', '--runs', type=int, default=1)

    args = parser.parse_args()

    tofl.utils.check_fractions(args.trn_frac, args.val_frac, args.tst_frac)
    tofl.utils.check_purities(args.purities)

    if args.queue:  # slurm cluster
        tag = tofl.utils.create_tag(args.purities, args.trn_frac)
        for run in range(args.runs):
            tofl.utils.submit_train_job(args.arch + '_topic', tag, run, args)
    else:
        train(args)
