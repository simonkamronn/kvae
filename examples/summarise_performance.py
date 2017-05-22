import os
import json
import numpy as np
import pandas as pd
import argparse


def main(logdir='.', sortby='p_smooth', output='.', filter=''):
    if output == 'logdir':
        output = logdir

    planning_log_file = 'imputation_results_missing_planning.npz'
    random_log_file = 'imputation_results_missing_random.npz'

    results = dict(config=list(), p_smooth=list(), p_gen=list(), p_filt=list(),
                   r_smooth=list(), r_gen=list(), r_filt=list())
    for subdir in os.listdir(logdir):
        if os.path.isfile(os.path.join(logdir, subdir, planning_log_file)):
            planning_results = np.load(os.path.join(logdir, subdir, planning_log_file))['results']

            # RNN results are formatted different
            if len(planning_results) == 4:
                idx = [1, 2, 3]
            else:
                idx = [0, 1, 2]

            results['p_smooth'].append(planning_results[idx[0]][0][5])
            results['p_filt'].append(planning_results[idx[1]][0][5])
            results['p_gen'].append(planning_results[idx[2]][0][5])

            random_results = np.load(os.path.join(logdir, subdir, random_log_file))['results']
            results['r_smooth'].append(random_results[idx[0]][0][5])
            results['r_filt'].append(random_results[idx[1]][0][5])
            results['r_gen'].append(random_results[idx[2]][0][5])

            results['config'].append(json.load(open(os.path.join(logdir, subdir, 'config.json'))))

    df = pd.DataFrame(results['config'])
    df['p_smooth'] = results['p_smooth']
    df['p_gen'] = results['p_gen']
    df['p_filt'] = results['p_filt']
    df['r_smooth'] = results['r_smooth']
    df['r_gen'] = results['r_gen']
    df['r_filt'] = results['r_filt']
    df['name'] = df['log_dir'].apply(os.path.basename)

    df = df.sort_values(by=sortby)

    if 'nogumbel' in filter:
        df = df[df['alpha_gumbel'] == False]
    elif 'gumbel' in filter:
        df = df[df['alpha_gumbel']]

    if 'rnn' in filter:
        df = df[df['alpha_rnn']]

    if 'ff' in filter:
        df = df[df['alpha_rnn'] == False]

    if 'nofifo' in filter:
        df = df[df['fifo_size'] == 1]
    elif 'fifo' in filter:
        df = df[df['fifo_size'] != 1]

    if 'baseline' in filter:
        df = df[df['name'].str.contains('aernn') | df['name'].str.contains('vrnn')]

    df.to_html(os.path.join(output, 'results.html'),
               columns=['p_smooth', 'p_filt', 'p_gen', 'r_smooth', 'r_filt', 'r_gen',
                        'name', 'alpha_gumbel', 'alpha_rnn', 'alpha_units', 'dim_a',
                        'dim_z', 'fifo_size', 'gumbel_decay_steps', 'gumbel_min', 'init_lr',
                        'kf_update_steps', 'noise_emission', 'noise_transition', 'num_epochs',
                        'only_vae_epochs', 'rnn_units', 'scale_reconstruction', 't_init_mask',
                        't_init_train_miss', 't_steps_mask', 'train_miss_prob', 'use_vae'])

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default='./logs')
    parser.add_argument("--sortby", type=str, default='p_smooth')
    parser.add_argument('--output', type=str, default='.')
    parser.add_argument('--filter', type=str, default='')
    args = parser.parse_args()

    main(args.logdir, args.sortby, args.output, args.filter)
