#!/usr/bin/env python3

import os
import numpy as np
import sys

from argparse import ArgumentParser
from energyflow import EFPSet
from glob import glob
from subprocess import call

def write(path, suffix, args):
    """
    Calculates the set of EFPs for each jet and stores as a numpy array per
    class.
    """

    # create EFP set
    efp_set = EFPSet(
        'd>0', f"d<={args.max_degree}", 'p==1', coords='ptyphim',
        check_input=False, beta=0.5
    )

    with np.load(path) as f:
        
        jets, labels = f['X'], f['y']
        quark_jets = jets[labels==1]
        gluon_jets = jets[labels==0]
        
        quark_path = os.path.join(
            args.savedir, f'd{args.max_degree}/quark_efps{suffix}.npz'
        )
        gluon_path = os.path.join(
            args.savedir, f'd{args.max_degree}/gluon_efps{suffix}.npz'
        )
        if args.overwrite or not os.path.exists(quark_path):
            print(f'Writing quark EFPs to {quark_path}')
            np.savez(
                quark_path, data=efp_set.batch_compute(quark_jets, n_jobs=8)
            )
        else:
            print(f'File {quark_path} already exists. Skipping')
        if args.overwrite or not os.path.exists(gluon_path):
            print(f'Writing gluon EFPs to {gluon_path}')    
            np.savez(
                gluon_path, data=efp_set.batch_compute(gluon_jets, n_jobs=8)
            )
        else:
            print(f'File {gluon_path} already exists. Skipping')

def submit(path, suffix, args):
    """
    Submits a job to the batch system that runs this script on the given
    file with the same configuration.
    """

    setup_cmd = 'ml foss/2021b python/3.9.6'
    drop_args = ['-q', '--queue', args.queue, '--paths']
    script_cmd = 'python ' + ' '.join(
        [a for a in sys.argv if a not in drop_args + args.paths]
        ) + f" --paths {path}"

    log_dir = os.path.join(args.savedir, 'log')
    os.makedirs(log_dir, exist_ok=True)
    cmd = (
        f"sbatch -p {args.queue} --mem 4G -N 1 -c {args.ncpus} "
        f"-t 24:00:00  -J write_efps{suffix} "
        f"-e {os.path.join(log_dir, '%x')}.err "
        f"-o {os.path.join(log_dir, '%x')}.out "
        f"--wrap \"{setup_cmd}; {script_cmd}\""
    )
    print(f'[CMD] {cmd}')
    if not args.dry:
        call(cmd, shell=True, executable='/bin/bash')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--paths', nargs='+', default=
        glob(os.path.expanduser('~/.energyflow/datasets/QG_jets*npz'))
    )
    parser.add_argument('-s', '--savedir', default=
        os.path.expanduser('~/.topicflow/efps')
    )
    parser.add_argument('-o', '--overwrite', action='store_true')
    parser.add_argument('-d', '--max_degree', type=int, default=4)
    parser.add_argument('-q', '--queue', default=None)
    parser.add_argument('-c', '--ncpus', type=int, default=8)
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args()
 
    if len(args.paths) == 0:
        print('No files found!')
        
    op = submit if args.queue else write
    for path in args.paths:
        suffix = os.path.basename(path).replace(
            'QG_jets', '').replace('.npz', '') or '_0'
        op(path, suffix, args)
