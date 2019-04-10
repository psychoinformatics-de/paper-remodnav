#!/usr/bin/env python

import os.path as op
import itertools
import numpy as np
import pylab as pl
import seaborn as sns
from remodnav import EyegazeClassifier
from glob import glob
#from remodnav.tests.test_labeled import load_data as load_anderson


def load_anderson(category, name):
    from scipy.io import loadmat
    from datalad.api import get
    from remodnav import clf as CLF
    import os.path as op

    fname = op.join(
        'remodnav', 'remodnav', 'tests', 'data', 'anderson_etal',
        'annotated_data', 'data used in the article',
        category, name + ('' if name.endswith('.mat') else '.mat'))
    get(fname)
    m = loadmat(fname)
    # viewing distance
    vdist = m['ETdata']['viewDist'][0][0][0][0]
    screen_width = m['ETdata']['screenDim'][0][0][0][0]
    screen_res = m['ETdata']['screenRes'][0][0][0][0]
    px2deg = CLF.deg_per_pixel(screen_width, vdist, screen_res)
    sr = float(m['ETdata']['sampFreq'][0][0][0][0])
    data = np.rec.fromarrays([
        m['ETdata']['pos'][0][0][:, 3],
        m['ETdata']['pos'][0][0][:, 4]],
        names=('x', 'y'))
    data[np.logical_and(data['x'] == 0, data['y'] == 0)] = (np.nan, np.nan)

    labels = m['ETdata']['pos'][0][0][:, 5]

    label_remap = {
        1: 'FIXA',
        2: 'SACC',
        3: 'PSO',
        4: 'PURS',
    }
    events = []
    ev_type = None
    ev_start = None
    for i in range(len(labels)):
        s = labels[i]
        if ev_type is None and s in label_remap.keys():
            ev_type = s
            ev_start = i
        elif ev_type is not None and s != ev_type:
            events.append(dict(
                id=len(events),
                label=label_remap.get(ev_type),
                start_time=0.0 if ev_start is None else
                float(ev_start) / sr,
                end_time=float(i) / sr,
            ))
            ev_type = s if s in label_remap.keys() else None
            ev_start = i
    if ev_type is not None:
        events.append(dict(
            id=len(events),
            label=label_remap.get(ev_type),
            start_time=0.0 if ev_start is None else
            float(ev_start) / sr,
            end_time=float(i) / sr,
        ))
    return data, labels, events, px2deg, sr


labeled_files = {
    'dots': [
        'TH20_trial1_labelled_{}.mat',
        'TH38_trial1_labelled_{}.mat',
        'TL22_trial17_labelled_{}.mat',
        'TL24_trial17_labelled_{}.mat',
        'UH21_trial17_labelled_{}.mat',
        'UH21_trial1_labelled_{}.mat',
        'UH25_trial1_labelled_{}.mat',
        'UH33_trial17_labelled_{}.mat',
        'UL27_trial17_labelled_{}.mat',
        'UL31_trial1_labelled_{}.mat',
        'UL39_trial1_labelled_{}.mat',
    ],
    'img': [
        'TH34_img_Europe_labelled_{}.mat',
        'TH34_img_vy_labelled_{}.mat',
        'TL20_img_konijntjes_labelled_{}.mat',
        'TL28_img_konijntjes_labelled_{}.mat',
        'UH21_img_Rome_labelled_{}.mat',
        'UH27_img_vy_labelled_{}.mat',
        'UH29_img_Europe_labelled_{}.mat',
        'UH33_img_vy_labelled_{}.mat',
        'UH47_img_Europe_labelled_{}.mat',
        'UL23_img_Europe_labelled_{}.mat',
        'UL31_img_konijntjes_labelled_{}.mat',
        'UL39_img_konijntjes_labelled_{}.mat',
        'UL43_img_Rome_labelled_{}.mat',
        'UL47_img_konijntjes_labelled_{}.mat',
    ],
    'video': [
        'TH34_video_BergoDalbana_labelled_{}.mat',
        'TH38_video_dolphin_fov_labelled_{}.mat',
        'TL30_video_triple_jump_labelled_{}.mat',
        'UH21_video_BergoDalbana_labelled_{}.mat',
        'UH29_video_dolphin_fov_labelled_{}.mat',
        'UH47_video_BergoDalbana_labelled_{}.mat',
        'UL23_video_triple_jump_labelled_{}.mat',
        'UL27_video_triple_jump_labelled_{}.mat',
        'UL31_video_triple_jump_labelled_{}.mat',
    ],
}


def get_durations(events, evcodes):
    events = [e for e in events if e['label'] in evcodes]
    # TODO minus one sample at the end?
    durations = [e['end_time'] - e['start_time'] for e in events]
    return durations


def print_duration_stats():
    for stimtype in ('img', 'dots', 'video'):
    #for stimtype in ('img', 'video'):
        for coder in ('MN', 'RA'):
            print(stimtype, coder)
            fixation_durations = []
            saccade_durations = []
            pso_durations = []
            purs_durations = []
            for fname in labeled_files[stimtype]:
                data, target_labels, target_events, px2deg, sr = load_anderson(
                    stimtype, fname.format(coder))
                fixation_durations.extend(get_durations(
                    target_events, ['FIXA']))
                saccade_durations.extend(get_durations(
                    target_events, ['SACC']))
                pso_durations.extend(get_durations(
                    target_events, ['PSO']))
                purs_durations.extend(get_durations(
                    target_events, ['PURS']))
            print(
                'FIX: %i (%i) [%i]' % (
                    np.mean(fixation_durations) * 1000,
                    np.std(fixation_durations) * 1000,
                    len(fixation_durations)))
            print(
                'SAC: %i (%i) [%i]' % (
                    np.mean(saccade_durations) * 1000,
                    np.std(saccade_durations) * 1000,
                    len(saccade_durations)))
            print(
                'PSO: %i (%i) [%i]' % (
                    np.mean(pso_durations) * 1000,
                    np.std(pso_durations) * 1000,
                    len(pso_durations)))
            print(
                'PURS: %i (%i) [%i]' % (
                    np.mean(purs_durations) * 1000,
                    np.std(purs_durations) * 1000,
                    len(purs_durations)))


def confusion(refcoder,
              coder,
              figures,
              stats):
    conditions = ['FIX', 'SAC', 'PSO', 'PUR']
    #conditions = ['FIX', 'SAC', 'PSO']
    label_map = {
        'FIXA': 'FIX',
        'FIX': 'FIX',
        'SACC': 'SAC',
        'ISAC': 'SAC',
        'HPSO': 'PSO',
        'IHPS': 'PSO',
        'LPSO': 'PSO',
        'ILPS': 'PSO',
        'PURS': 'PUR',
    }
    anderson_remap = {
        'FIX': 1,
        'SAC': 2,
        'PSO': 3,
        'PUR': 4,
    }
    plotter = 1
    # coders are in axis labels too
    #pl.suptitle('Jaccard index for movement class labeling {} vs. {}'.format(
    #    refcoder, coder))
    for stimtype in ('img', 'dots', 'video'):
        conf = np.zeros((len(conditions), len(conditions)), dtype=float)
        jinter = np.zeros((len(conditions), len(conditions)), dtype=float)
        junion = np.zeros((len(conditions), len(conditions)), dtype=float)
        for fname in labeled_files[stimtype]:
            labels = []
            data = None
            px2deg = None
            sr = None
            for src in (refcoder, coder):
                if src in ('RA', 'MN'):
                    data, target_labels, target_events, px2deg, sr = \
                        load_anderson(stimtype, fname.format(src))
                    labels.append(target_labels.astype(int))
                else:
                    clf = EyegazeClassifier(
                        px2deg=px2deg,
                        sampling_rate=sr,
                        pursuit_velthresh=5.,
                        noise_factor=3.0,
                        lowpass_cutoff_freq=10.0,
                        min_fixation_duration=0.055,
                    )
                    p = clf.preproc(data)
                    events = clf(p)

                    # convert event list into anderson-style label array
                    l = np.zeros(labels[0].shape, labels[0].dtype)
                    for ev in events:
                        l[int(ev['start_time'] * sr):int((ev['end_time']) * sr)] = \
                            anderson_remap[label_map[ev['label']]]
                    labels.append(l)

            nlabels = [len(l) for l in labels]
            if len(np.unique(nlabels)) > 1:
                print(
                    "% #\n% # %INCONSISTENCY Found label length mismatch "
                    "between coders ({}, {}) for: {}\n% #\n".format(
                        refcoder, coder, fname))
                print('% Truncate labels to shorter sample: {}'.format(
                    nlabels))
                order_idx = np.array(nlabels).argsort()
                labels[order_idx[1]] = \
                    labels[order_idx[1]][:len(labels[order_idx[0]])]

            for c1, c1label in enumerate(conditions):
                for c2, c2label in enumerate(conditions):
                    intersec = np.sum(np.logical_and(
                        labels[0] == anderson_remap[c1label],
                        labels[1] == anderson_remap[c2label]))
                    union = np.sum(np.logical_or(
                        labels[0] == anderson_remap[c1label],
                        labels[1] == anderson_remap[c2label]))
                    jinter[c1, c2] += intersec
                    junion[c1, c2] += union
                    #if c1 == c2:
                    #    continue
                    conf[c1, c2] += np.sum(np.logical_and(
                        labels[0] == anderson_remap[c1label],
                        labels[1] == anderson_remap[c2label]))

        nsamples = np.sum(conf)
        nsamples_nopurs = np.sum(conf[:3, :3])
        # zero out diagonal for bandwidth
        conf *= ((np.eye(len(conditions)) - 1) * -1)
        if figures:
            pl.subplot(1, 3, plotter)
            sns.heatmap(
                #(conf / nsamples) * 100,
                jinter / junion,
                square=True,
                annot=True,
                cmap=sns.cm.rocket_r,
                xticklabels=conditions,
                yticklabels=conditions,
                vmin=0.0,
                vmax=1.0,
            )
            pl.xlabel('{} labeling'.format(refcoder))
            pl.ylabel('{} labeling'.format(coder))
            # stats are given proper below
            #pl.title('"{}" (glob. misclf-rate): {:.1f}% (w/o pursuit: {:.1f}%)'.format(
            #    stimtype,
            #    (np.sum(conf) / nsamples) * 100,
            #    (np.sum(conf[:3, :3]) / nsamples_nopurs) * 100))
            pl.title(stimtype)
            plotter += 1
        msclf_refcoder = dict(zip(conditions, conf.sum(axis=1)/conf.sum() * 100))
        msclf_coder = dict(zip(conditions, conf.sum(axis=0)/conf.sum() * 100))

        if stats:
        # print results as LaTeX commands
            label_prefix = '{}{}{}'.format(stimtype, refcoder, coder)
            for key, format, value in (
                    ('MCLF', '%.1f', (np.sum(conf) / nsamples) * 100),
                    ('MclfWOP', '%.1f',
                        (np.sum(conf[:3, :3]) / nsamples_nopurs) * 100),
                    ('FIXref', '%.0f', msclf_refcoder['FIX']),
                    ('SACref', '%.0f', msclf_refcoder['SAC']),
                    ('PSOref', '%.0f', msclf_refcoder['PSO']),
                    ('SPref', '%.0f', msclf_refcoder['PUR']),
                    ('FIXcod', '%.0f', msclf_coder['FIX']),
                    ('SACcod', '%.0f', msclf_coder['SAC']),
                    ('PSOcod', '%.0f', msclf_coder['PSO']),
                    ('SPcod', '%.0f', msclf_coder['PUR'])):
                print('\\newcommand{\\%s%s}{%s}'
                      % (label_prefix, key, format % value))

            # print original outputs, but make them LaTeX-safe with '%'. This
            # should make it easier to check correct placements of stats in the
            # table
            print('% ### {}'.format(stimtype))
            print('% Comparison | MCLF | MCLFw/oP | Method | Fix | Sacc | PSO | SP')
            print('% --- | --- | --- | --- | --- | --- | --- | ---')
            print('% {} v {} | {:.1f} | {:.1f} | {} | {:.0f} | {:.0f} | {:.0f} | {:.0f}'.format(
                refcoder,
                coder,
                (np.sum(conf) / nsamples) * 100,
                (np.sum(conf[:3, :3]) / nsamples_nopurs) * 100,
                refcoder,
                msclf_refcoder['FIX'],
                msclf_refcoder['SAC'],
                msclf_refcoder['PSO'],
                msclf_refcoder['PUR'],
            ))
            print('% -- | --  | -- | {} | {:.0f} | {:.0f} | {:.0f} | {:.0f}'.format(
                coder,
                msclf_coder['FIX'],
                msclf_coder['SAC'],
                msclf_coder['PSO'],
                msclf_coder['PUR'],
        ))


def savefigs(fig,
             stat):
    """
    small helper function to save all confusion matrices
    """

    for pair in itertools.combinations(['MN', 'RA', 'ALGO'], 2):
        pl.figure(
            # fake size to get the font size down in relation
            figsize=(14, 3),
            dpi=120,
            frameon=False)
        confusion(pair[0],
                  pair[1],
                  fig,
                  stat)
        pl.savefig(
            op.join('img', 'confusion_{}_{}.svg'.format(*pair)),
            transparent=True,
            bbox_inches="tight")
        pl.close()


def savegaze():
    """
    small function to generate and save remodnav classification figures
    """
    from remodnav.tests import utils as ut
    import pylab as pl
    import datalad.api as dl

    # use two examplary files (lab + MRI) used during testing as well
    # hardcoding those, as I see no reason for updating them
    infiles = [
        op.join(
            'data', 'studyforrest-data-eyemovementlabels', 'inputs',
            'raw_eyegaze', 'sub-32', 'beh',
            'sub-32_task-movie_run-2_recording-eyegaze_physio.tsv.gz'),
        op.join(
            'data', 'studyforrest-data-eyemovementlabels', 'inputs',
            'raw_eyegaze', 'sub-09', 'ses-movie',  'func',
            'sub-09_ses-movie_task-movie_run-2_recording-eyegaze_physio.tsv.gz'
        ),
    ]
    dl.get(infiles)
    for f in infiles:
        # read data
        data = np.recfromcsv(f,
                             delimiter='\t',
                             names=['x', 'y', 'pupil', 'frame'])

        # adjust px2deg conversion factor according to datafile
        pxdeg, ext = (0.0266711972026, 'lab') if '32' in f \
            else (0.0185581232561, 'mri')
        clf = EyegazeClassifier(
            px2deg=pxdeg,
            sampling_rate=1000.0)
        p = clf.preproc(data)
        # lets go with 10 seconds to actually see details. This particular time
        # window is within the originally plotted 50s and contains missing data
        # for both data types (lab & mri)
        events = clf(p[30000:40000])

        fig = pl.figure(
            # fake size to get the font size down in relation
            figsize=(14, 2),
            dpi=120,
            frameon=False)
        ut.show_gaze(
            #data[30000:40000],
            pp=p[30000:40000],
            events=events,
            sampling_rate=1000.0,
            show_vels=True)
        pl.savefig(
            op.join('img', 'remodnav_{}.svg'.format(ext)),
            transparent=True,
            bbox_inches="tight")
        pl.close()


def mainseq(s_mri = 'sub-19',
            s_lab = 'sub-29'):
    """
    plot main sequences from movie data for lab and mri subjects.
    """
    import pandas as pd
    from matplotlib.lines import Line2D

    mris = ['01', '02', '03', '04', '05', '06', '09', '10', '14', '15', '16', '17', '18', '19', '20']
    labs = ['22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36']

    datapath = op.join('data',
                       'studyforrest-data-eyemovementlabels',
                       'sub*',
                       '*.tsv')
    data = sorted(glob(datapath))

    # create dataframes for mri and lab subjects to plot in seperate plots
    mri_dfs = []
    lab_dfs = []

    for f in data[:120]:
        assert [mri in f for mri in mris]
        mri_dfs.append(pd.read_csv(f, header=0, delim_whitespace=True))
    mri_df = pd.concat(mri_dfs)

    for f in data[120:]:
        assert [lab in f for lab in labs]
        lab_dfs.append(pd.read_csv(f, header=0, delim_whitespace=True))
    lab_df = pd.concat(lab_dfs)

    # also create a dataframe for an individual subjects run
    sub_mri = op.join('data',
                      'studyforrest-data-eyemovementlabels',
                      s_mri,
                      '{}_task-movie_run-5_events.tsv'.format(s_mri))
    sub_mri_df = pd.read_csv(sub_mri, header=0, delim_whitespace=True)

    sub_lab = op.join('data',
                      'studyforrest-data-eyemovementlabels',
                      s_lab,
                      '{}_task-movie_run-5_events.tsv'.format(s_lab))
    sub_lab_df = pd.read_csv(sub_lab, header=0, delim_whitespace=True)

    for (df, ext) in [(mri_df, 'mri'),
                      (lab_df, 'lab'),
                      (sub_mri_df, 'sub_mri'),
                      (sub_lab_df, 'sub_lab')]:

        # extract relevant event types
        SACCs = df[df.label == 'SACC']
        ISACs = df[df.label == 'ISAC']
        HPSOs = df[(df.label == 'HPSO') | (df.label == 'IHPS')]
        LPSOs = df[(df.label == 'LPSO') | (df.label == 'ILPS')]

        fig = pl.figure(
            # fake size to get the font size down in relation
            figsize=(14, 5),
            dpi=120,
            frameon=False)

        for ev, sym, color in (
                (ISACs, '.', 'darkred'),
                (SACCs, '.', 'red'),
                (HPSOs, '+', 'dodgerblue'),
                (LPSOs, '+', 'darkblue'))[::-1]:
            pl.loglog(
                ev['amp'],
                ev['peak_vel'],
                sym,
                alpha=0.20,
                color=color,
                lw = 1
            )

        # cheat: custom legend to not propagate alpha into legend markers
        custom_legend = [Line2D([0], [0],
                                marker='.',
                                color='w',
                                markerfacecolor='darkred',
                                label='Saccade (ISAC)',
                                markersize=10),
                         Line2D([0], [0],
                                marker='.',
                                color='w',
                                markerfacecolor='red',
                                label='Major saccade (SACC)',
                                markersize=10),
                         Line2D([0], [0],
                                marker='P',
                                color='w',
                                markerfacecolor='dodgerblue',
                                label='High velocity PSOs',
                                markersize=10),
                         Line2D([0], [0],
                                marker='P',
                                color='w',
                                markerfacecolor='darkblue',
                                label='Low velocity PSOs',
                                markersize=10)]

        pl.ylim((10.0, 1000))
        pl.xlim((0.01, 40.0))
        pl.legend(handles=custom_legend, loc=4)
        pl.ylabel('peak velocities (deg/s)')
        pl.xlabel('amplitude (deg)')
        pl.savefig(
            op.join('img', 'mainseq_{}.svg'.format(ext)),
            transparent=True,
            bbox_inches="tight")
        pl.close()


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--figure', help='if given, figures will be produced.',
        action='store_true', default=False)
    parser.add_argument(
        '-s', '--stats', help='if given, stats will be produced to stdout',
        action='store_true', default=False)
    parser.add_argument(
        '-r', '--remodnav',
        help='if given, remodnav classification figures are produced.',
        action='store_true', default=False)

    args = parser.parse_args()
    # generate & save figures; export the misclassification stats
    if args.figure or args.stats:
        savefigs(args.figure, args.stats)
    if args.remodnav:
        savegaze()
