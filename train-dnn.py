#!/usr/bin/env python3

def train(args):

    import data
    import nn
    import numpy as np
    import pickle
    import topicflow as tpf
    from tensorflow.config.threading import set_inter_op_parallelism_threads
    from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l1

    # set threads
    set_inter_op_parallelism_threads(4)

    # log arguments
    print(vars(args))

    # initialize classifier
    dnn_config = {
        'hidden_units': [100]*5,
        'output_dim': 2,
        'output_activation': 'softmax',
        'regularizer': (
            None if not args.regularization else l1(float(args.regularization))
        ),
        'dropout': None if not args.dropout else float(args.dropout),
    }
    dnn = nn.make_dnn(**dnn_config)
    dnn.build([None, args.dimension])
    dnn.summary()
    dnn.compile(
        optimizer=Adam(args.learning_rate), loss='bce', metrics=['AUC', 'acc']
    )

    # construct datasets
    fractions = {
        'trn': args.trn_frac, 'val': args.val_frac, 'tst': args.tst_frac
    }
    datasets = data.get_mixture_datasets(
        fractions=fractions, purities=args.purities, dim=args.dimension,
        batch_size=args.batch_size, pca=not args.nopca,
        efp_dir=args.efp_dir or tpf.data.EFP_DIR
    )
    # fit model to train data
    tensorboard = TensorBoard(
        log_dir=os.path.join(args.savedir, 'summary'), write_graph=False,
        update_freq='epoch', profile_batch=0
    )
    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=1e-4, patience=args.patience,
        restore_best_weights=True, verbose=1
    )
    dnn.fit(
        datasets['trn'], validation_data=datasets['val'], epochs=args.epochs,
        callbacks=[tensorboard, early_stopping], verbose=2
    )

    # evaluate model on PURE test data
    if args.purities[0] !=0 and args.purities[1] != 1:
        pure_test_dataset = data.get_mixture_datasets(
            fractions=fractions, purities=[1., 0.], batch_size=5000,
            dim=args.dimension, pca=not args.nopca,
            efp_dir=args.efp_dir or tpf.data.EFP_DIR
        )['tst']
    else:
        pure_test_dataset = datasets['tst']

    metrics = dnn.evaluate(pure_test_dataset)
    
    # extract topic fractions from model predictions
    qpreds, gpreds = [], []
    for x, y in datasets['trn']:
        preds = dnn.predict_on_batch(x)
        qpreds = np.append(qpreds, preds[y[:,1]==1, 1])
        gpreds = np.append(gpreds, preds[y[:,1]==0, 1])
    demix_dict = tpf.utils.demix_topics(qpreds, gpreds, 64, 6)

    # save model and test metrics
    weights_path = os.path.join(args.savedir, 'weights')
    metrics_path = os.path.join(args.savedir, 'metrics.p')
    demix_path = os.path.join(args.savedir, 'demix_dict.p')
    print(f'Saving weights to {weights_path}')
    print(f'Saving metrics to {metrics_path}')
    print(f'Saving topic fractions and reducibilities to {demix_path}')
    if not args.dry:
        dnn.save_weights(weights_path)
        with open(metrics_path, 'wb') as fm, open(demix_path, 'wb') as fd:
            pickle.dump(metrics, fm)
            pickle.dump(demix_dict, fd)

if __name__ == '__main__':

    import os
    import topicflow as tpf
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-P', '--purities', type=float, nargs=2, required=True)
    parser.add_argument('-D', '--dimension', type=int, required=True)
    parser.add_argument('--nopca', action='store_true')
    parser.add_argument('-s', '--savedir', default=os.getcwd())
    parser.add_argument('-b', '--batch_size', type=int, default=500)
    parser.add_argument('-e', '--epochs', type=int, default=200)
    parser.add_argument('-p', '--patience', type=int, default=10)
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-r', '--regularization', default=None)
    parser.add_argument('-d', '--dropout', default=0.15)
    parser.add_argument('-T', '--trn_frac', type=float, default=0.75)
    parser.add_argument('-V', '--val_frac', type=float, default=0.1)
    parser.add_argument('-E', '--tst_frac', type=float, default=0.15)
    parser.add_arguemtn('--efp_dir', default=None)
    parser.add_argument('--dry', action='store_true')
    
    parser.add_argument('-q', '--queue', default=None)
    parser.add_argument('-n', '--runs', type=int, default=1)
    parser.add_argument('-m', '--memory', default='4G')
    parser.add_argument('-t', '--time', default='3:00:00')
    args = parser.parse_args()

    tpf.utils.check_fractions(args.trn_frac, args.val_frac, args.tst_frac)
    tpf.utils.check_purities(args.purities)
    
    args.purities = sorted(args.purities)
    tag = tpf.utils.create_tag(args.purities, args.train_frac)
    if args.queue:
        for run in range(args.runs):
            tpf.utils.submit_train_job('dnn', tag, run, args)
    else:
        train(args)
