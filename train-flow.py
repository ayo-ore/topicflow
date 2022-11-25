#!/usr/bin/env python3

def train(args):

    import sys
    sys.path.append('/data/projects/punim0011/aore/weak')

    import numpy as np
    import pickle
    from sklearn.metrics import roc_auc_score
    from tensorflow.config.threading import set_inter_op_parallelism_threads
    from tensorflow.keras import callbacks, optimizers, regularizers

    # set threads
    set_inter_op_parallelism_threads(4)

    # log arguments
    print(vars(args))

    # construct datasets
    fractions = {
        'trn': args.trn_frac, 'val': args.val_frac, 'tst': args.tst_frac
    }

    f1, f2 = sorted(args.purities, reverse=True)
    datasets = tpf.get_mixture_datasets(
        purities=[f1,f2], fractions=fractions, dim=args.dimension,
        pca=not args.nopca, batch_size=args.batch_size,
        efp_dir=args.efp_dir or tpf.data.EFP_DIR
    )

    model_config = {
        'dim': args.dimension,
        'bijector_config': {
            'final_time': 1.0,
            'atol': args.ode_atol,
            'exact_trace': False
        },
        'net_config': {
            'num_residual_blocks': args.ode_num_residual_blocks,
            'hidden_dim': args.ode_hidden_dim,
            'num_block_layers': args.ode_num_block_layers,
            'activation': 'tanh',
            'regularizer': regularizers.l1(
                args.ode_weight_decay
            ) if args.ode_weight_decay else None,
            'dropout': args.ode_dropout
        }
    } if args.arch == 'node' else {
        'dim': args.dimension,
        'num_transform_layers': args.num_transform_layers,
        'batchnorm': False,
        'bijector_config': {'fraction_masked': 0.5},
        'net_config': {
            'layer_sizes': args.spline_layer_sizes,
            'num_knots': args.spline_num_knots,
            'boundary': [-args.spline_boundary, args.spline_boundary],
            'min_width': args.spline_min_width,
            'min_slope': args.spline_min_slope,
            'regularizer': regularizers.l2(
                args.spline_regularization
            ) if args.spline_regularization else None,
            'dropout': args.spline_dropout
        }
    } if args.arch == 'rqs' else None

    arch = tpf.NeuralODEFlow if args.arch == 'node' else tpf.RQSCouplingFlow
    flow1 = tpf.TopicFlow(args.dimension, (1-f1)/(1-f2), 0, arch, model_config)
    flow2 = tpf.TopicFlow(args.dimension,         f2/f1, 1, arch, model_config)
    flow1.build([1, args.dimension])
    flow2.build([1, args.dimension])
   
    flow1.compile(optimizers.Adam(
        args.learning_rate, clipvalue=5 if args.arch == 'rqs' else None)
    )
    flow2.compile(optimizers.Adam(
        args.learning_rate, clipvalue=5 if args.arch == 'rqs' else None)
    )

    # callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', min_delta=1e-4, patience=args.patience, verbose=1,
        restore_best_weights=True,
    )
    kappa_warmup = tpf.utils.ZeroWarmUp(
        param_name='kappa', warmup_epochs=args.warmup_epochs
    )
    reduce_on_plateau = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=5,
        verbose=1, min_lr=1e-6
    )
    tensorboard1 = callbacks.TensorBoard(
        log_dir=os.path.join(args.savedir, 'summary/topic1'), write_graph=False,
        update_freq='epoch', profile_batch=0
    )
    tensorboard2 = callbacks.TensorBoard(
        log_dir=os.path.join(args.savedir, 'summary/topic2'), write_graph=False,
        update_freq='epoch', profile_batch=0
    )
    
    # training
    flow1.fit(
        datasets['trn'], validation_data=datasets['val'].unbatch().batch(5000),
        epochs=args.epochs, verbose=2, callbacks=[
            early_stopping, kappa_warmup, reduce_on_plateau, tensorboard1
        ]
    )
    flow2.fit(
        datasets['trn'], validation_data=datasets['val'].unbatch().batch(5000),
        epochs=args.epochs, verbose=2, callbacks=[
            early_stopping, kappa_warmup, reduce_on_plateau, tensorboard2
        ]
    )

    # save model
    weights_path1 = os.path.join(args.savedir, 'weights1')
    weights_path2 = os.path.join(args.savedir, 'weights2')
    model_config_path = os.path.join(args.savedir, 'model_config.p')
    if not args.dry:
        print(f'Saving weights to {weights_path1}, {weights_path2}')
        flow1.flow.save_weights(weights_path1)
        flow2.flow.save_weights(weights_path2)
        with open(model_config_path, 'wb') as mf:
            print(f'Saving model config to {model_config_path}')
            pickle.dump(model_config, mf)

    # test model
    test_data = [
        np.vstack(list(tpf.get_mixture_datasets(
            purities=[p], dim=args.dimension, batch_size=args.batch_size,
            labels=False, efp_dir = args.efp_dir or tpf.data.EFP_DIR,
            fractions={
                'trn': fractions['trn'],
                'val': 0.85-fractions['trn'],
                'tst': 0.15
            }
        )['tst'])) for p in [1., 0.]
    ]

    # wassersteins
    test_samples = [
        flow1.flow.sample(150_000).numpy(), flow2.flow.sample(150_000).numpy()
    ]
    test_distances = [
        tpf.utils.np_wasserstein(d.T, s.T)
        for d, s in zip(test_data, test_samples)
    ]
    distance_path = os.path.join(args.savedir, 'wassersteins.p')
    print(f'Saving wasserstein distances to {distance_path}.')
    if not args.dry:
        with open(distance_path, 'wb') as f:
            pickle.dump(test_distances, f)

    # generative classification metrics
    preds = np.hstack([flow1(d) - flow2(d) for d in test_data])
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

    import os
    import topicflow as tpf
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-P', '--purities', type=float, nargs=2, required=True)
    parser.add_argument('-D', '--dimension', type=int, required=True)
    parser.add_argument('--nopca', action='store_true')
    parser.add_argument('-s', '--savedir', default=os.getcwd())
    parser.add_argument('--efp_dir', default=None)
    parser.add_argument('--dry', action='store_true')

    parser.add_argument('-b', '--batch_size', type=int, default=1000)
    parser.add_argument('-e', '--epochs', type=int, default=200)
    parser.add_argument('-p', '--patience', type=int, default=10)
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-w', '--warmup_epochs', type=int, default=0)
    parser.add_argument('-T', '--trn_frac', type=float, default=0.75)
    parser.add_argument('-V', '--val_frac', type=float, default=0.1)
    parser.add_argument('-E', '--tst_frac', type=float, default=0.15)

    parser.add_argument('--arch', choices=('node', 'rqs'), default='node')
    parser.add_argument('-L', '--num_transform_layers', type=int, default=16)
    parser.add_argument('--spline_layer_sizes', type=int, nargs='+', default=[128]*2)
    parser.add_argument('--spline_num_knots', type=int, default=8)
    parser.add_argument('--spline_boundary', type=int, default=6)
    parser.add_argument('--spline_min_width', type=float, default=1e-4)
    parser.add_argument('--spline_min_slope', type=float, default=1e-4)
    parser.add_argument('--spline_regularization', type=float, default=0.)
    parser.add_argument('--spline_dropout', type=float, default=None)

    parser.add_argument('--ode_atol', type=float, default=1e-6)
    parser.add_argument('--ode_num_residual_blocks', type=int, default=2)
    parser.add_argument('--ode_hidden_dim', type=int, default=256)
    parser.add_argument('--ode_num_block_layers', type=float, default=2)
    parser.add_argument('--ode_weight_decay', type=float, default=0.)
    parser.add_argument('--ode_dropout', type=float, default=None)

    parser.add_argument('-q', '--queue', default=None)
    parser.add_argument('-m', '--memory', default='48G')
    parser.add_argument('-t', '--time', default='48:00:00')
    parser.add_argument('-n', '--runs', type=int, default=1)

    args = parser.parse_args()

    tpf.utils.check_fractions(args.trn_frac, args.val_frac, args.tst_frac)
    tpf.utils.check_purities(args.purities)

    if args.queue: # slurm cluster
        tag = tpf.utils.create_tag(args.purities, args.train_frac)
        for run in range(args.runs):
            tpf.utils.submit_train_job(args.arch +'_topic', tag, run, args)
    else:
        train(args)
