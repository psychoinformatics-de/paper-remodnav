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

    fname = op.join(*(
        ('remodnav', 'remodnav', 'tests', 'data', 'anderson_etal',
         'annotated_data') + \
        (('fix_by_Zemblys2018',)
         if name  == 'UH29_img_Europe_labelled_FIX_MN.mat'
         else ('data used in the article', category)
        ) + \
        (name + ('' if name.endswith('.mat') else '.mat'),))
    )
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


#make label_map global - we need it twice
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


# we need the distribution parameters of all algorithms and human coders
# in tables 3, 4, 5, 6 from Andersson et al., 2017. Well worth double-checking,
# I needed to hand-copy-paste from the paper. The summary statistics were made
# publicly available in the file # matlab_analysis_code/20150807.mat in the
# original authors GitHub repository
# (https://github.com/richardandersson/EyeMovementDetectorEvaluation/blob/0e6f82708e10b48039763aa1078696e802260674/matlab_analysis_code/20150807.mat).
# The first two entries within each value-list belong to human coders

image_params = {
    "FIX": {
        'alg': ['MN', 'RA', 'CDT', 'IDT', 'IKF', 'IMST', 'IHMM', 'IVT', 'NH', 'BIT'],
        'mn': [248, 242, 397, 399, 174, 304, 133, 114, 258, 209],
        'sd': [271, 273, 559, 328, 239, 293, 216, 204, 299, 136],
        'no': [380, 369, 251, 242, 513, 333, 701, 827, 292, 423]
    },
    "SAC": {
        'alg': ['MN', 'RA', 'EM', 'IDT', 'IKF', 'IMST', 'IHMM', 'IVT', 'NH', 'LNS'],
        'mn': [30, 31, 25, 35, 62, 17, 48, 41, 50, 29],
        'sd': [17, 15, 22, 15, 37, 10, 26, 22, 20, 12],
        'no': [376, 372, 787, 258, 353, 335, 368, 373, 344, 390]
    },
    "PSO" : {
        'alg': ['MN', 'RA', 'NH', 'LNS'],
        'mn': [21, 21, 28, 25],
        'sd': [11, 9, 13, 9],
        'no': [312, 309, 237, 319]
    },
    "PUR": {
        'alg': ['MN', 'RA'],
        'mn': [363, 305],
        'sd': [187, 184],
        'no': [3, 16]
    }
}

dots_params = {
    "FIX": {
        'alg': ['MN', 'RA', 'CDT', 'IDT', 'IKF', 'IMST', 'IHMM', 'IVT', 'NH', 'BIT'],
        'mn': [161, 131, 60, 323, 217, 268, 214, 203, 380, 189],
        'sd': [30, 99, 127, 146, 184, 140, 286, 282, 333, 113],
        'no': [2, 13, 165, 8, 72, 12, 67, 71, 30, 67]
    },
    "SAC": {
        'alg': ['MN', 'RA', 'EM', 'IDT', 'IKF', 'IMST', 'IHMM', 'IVT', 'NH', 'LNS'],
        'mn': [23, 22, 17, 32, 60, 13, 41, 36, 43, 26],
        'sd': [10, 11, 14, 14, 26, 5, 17, 14, 16, 11],
        'no': [47, 47, 93, 10, 29, 18, 27, 28, 42, 53]
    },
    "PSO" : {
        'alg': ['MN', 'RA', 'NH', 'LNS'],
        'mn': [15, 15, 24, 20],
        'sd': [5, 8, 12, 9],
        'no': [33, 28, 17, 31]
    },
    "PUR": {
        'alg': ['MN', 'RA'],
        'mn': [375, 378],
        'sd': [256, 364],
        'no': [37, 33]
    }
}

video_params = {
    "FIX": {
        'alg': ['MN', 'RA', 'CDT', 'IDT', 'IKF', 'IMST', 'IHMM', 'IVT', 'NH', 'BIT'],
        'mn': [318, 240, 213, 554, 228, 526, 234, 202, 429, 248],
        'sd': [289, 189, 297, 454, 296, 825, 319, 306, 336, 215],
        'no': [67, 67, 211, 48, 169, 71, 194, 227, 83, 170]
    },
    "SAC": {
        'alg': ['MN', 'RA', 'EM', 'IDT', 'IKF', 'IMST', 'IHMM', 'IVT', 'NH', 'LNS'],
        'mn': [26, 25, 20, 24, 55, 18, 42, 36, 44, 28],
        'sd': [13, 12, 16, 53, 20, 10, 18, 16, 18, 12],
        'no': [116, 126, 252, 41, 107, 76, 109, 112, 1104, 122]
    },
    "PSO" : {
        'alg': ['MN', 'RA', 'NH', 'LNS'],
        'mn': [20, 17, 28, 24],
        'sd': [11, 8, 13, 10],
        'no': [97, 89, 78, 87]
    },
    "PUR": {
        'alg': ['MN', 'RA'],
        'mn': [521, 472],
        'sd': [347, 319],
        'no': [50, 68]
    }
}


mri_ids = ['01', '02', '03', '04', '05', '06', '09', '10', '14', '15',
           '16', '17', '18', '19', '20']
lab_ids = ['22', '23', '24', '25', '26', '27', '28', '29', '30', '31',
           '32', '33', '34', '35', '36']


# this used to be within confusion(), is global now because we also need it for Kappa()
# --> defines mapping between remodnav labels (strings) and andersson labels (ints)
anderson_remap = {
    'FIX': 1,
    'SAC': 2,
    'PSO': 3,
    'PUR': 4,
}


def get_durations(events, evcodes):
    events = [e for e in events if e['label'] in evcodes]
    # TODO minus one sample at the end?
    durations = [e['end_time'] - e['start_time'] for e in events]
    return durations


def confusion(refcoder,
              coder,
              figures,
              stats):
    conditions = ['FIX', 'SAC', 'PSO', 'PUR']
    #conditions = ['FIX', 'SAC', 'PSO']
    plotter = 1
    # initialize a maximum misclassification rate, to later automatically reference,
    max_mclf = 0
    # coders are in axis labels too
    #pl.suptitle('Jaccard index for movement class labeling {} vs. {}'.format(
    #    refcoder, coder))
    for stimtype, stimlabel in (
            ('img', 'Images'),
            ('dots', 'Dots'),
            ('video', 'Videos')):
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
                    finame = fname.format(src)
                    if finame == 'UH29_img_Europe_labelled_MN.mat':
                        # pick up Zemblys fix
                        finame = 'UH29_img_Europe_labelled_FIX_MN.mat'
                    data, target_labels, target_events, px2deg, sr = \
                        load_anderson(stimtype, finame)
                    labels.append(target_labels.astype(int))
                else:
                    clf = EyegazeClassifier(
                        px2deg=px2deg,
                        sampling_rate=sr,
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
            pl.title(stimlabel)
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
                # update classification performance only if there sth worse
                if (np.sum(conf[:3, :3]) / nsamples_nopurs * 100) > max_mclf:
                    max_mclf = (np.sum(conf[:3, :3]) / nsamples_nopurs * 100)
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
    return max_mclf


def savefigs(fig,
             stat):
    """
    small helper function to save all confusion matrices
    """
    max_mclf = 0
    for pair in itertools.combinations(['MN', 'RA', 'AL'], 2):
        pl.figure(
            # fake size to get the font size down in relation
            figsize=(14, 3),
            dpi=120,
            frameon=False)
        cur_max_mclf = confusion(pair[0],
                                 pair[1],
                                 fig,
                                 stat)
        pl.savefig(
            op.join('img', 'confusion_{}_{}.svg'.format(*pair)),
            transparent=True,
            bbox_inches="tight")
        pl.close()
        if cur_max_mclf > max_mclf:
            max_mclf = cur_max_mclf
    if stat:
        print('\\newcommand{\\maxmclf}{%s}'
              % ('%.1f' % max_mclf))


def quality_stats():
    """
    Computes the percent of signal loss in raw data
    Note: takes a while to run (30 subj. x 8 runs x 15 min rec with 1000Hz sr),
    therefore, I'm just adding results here for now
    and save the resulting histogram to the repository.
    \newcommand{\avglosslab}{0.041005777923189775}
    \newcommand{\avglossmri}{0.1507901497174581}

    To include this computation in a run with Make, add this function to the
    list of command invocations if the script is ran from the command line at the
    end of the script.
    """
    import datalad.api as dl
    import matplotlib.pyplot as plt

    datapath_mri = op.join('data', 'raw_eyegaze', 'sub-*', 'ses-movie', 'func',
                           'sub-*_ses-movie_task-movie_run-*_recording-eyegaze_physio.tsv.gz')
    datapath_lab = op.join('data', 'raw_eyegaze', 'sub-*', 'beh',
                           'sub-*_task-movie_run-*_recording-eyegaze_physio.tsv.gz')

    for (data, assoc) in [(datapath_lab, 'lab'),
                          (datapath_mri, 'mri')]:
        infiles = glob(data)
        for f in infiles:
            dl.get(f)
        # make sure we have 15 subjects' data
        assert len(infiles) == 120
        print("Currently processing data from {} sample".format(assoc))
        # set sampling rate and px2deg
        px2deg = 0.0266711972026 if assoc == 'lab' else 0.0185581232561
        sr = 1000
        # calculate percent signal loss across subjects and runs
        losses = []
        vels = []
        for f in infiles:
            data = np.recfromcsv(f,
                                 delimiter='\t',
                                 names=['x', 'y', 'pupil', 'frame'])
            # all periods of signal loss are marked as nan in the data
            signal_loss = np.sum(np.isnan(data['x'])) / len(data['x'])
            losses.append(signal_loss)
            velocities = cal_velocities(data=data,
                                        px2deg=px2deg,
                                        sr=sr)
            vels.append(velocities)
        print("Calculated velocities and losses for {} sample".format(assoc))
        # average across signal losses in sample (mri or lab)
        loss = np.nanmean(losses)
        # print results as Latex command using 'assoc' as sample identifier in name
        label_loss = 'avgloss{}'.format(assoc)
        print('\\newcommand{\\%s}{%s}'
              % (label_loss, loss))
        # vels is a list of arrays atm
        v = np.concatenate(vels).ravel()
        if assoc == 'lab':
            v_lab = v
        elif assoc == 'mri':
            v_mri = v

    # plot velocities in a histogram on logscale
    # create non-linear non-equal bin sizes, as x axis will be log
    hist, bins, _ = plt.hist(v[~np.isnan(v)], bins=40)
    plt.close()
    logbins = np.logspace(1,    # don't start with 0, does not make sense in logspace
                          np.log10(bins[-1]),
                          len(bins))
    fig, ax = plt.subplots()
    fig.set_figheight(3)
    fig.set_figwidth(5)
    ax.set_ylabel('frequency')
    ax.set_xlabel('velocities (deg/s)')
    plt.hist(v_mri[~np.isnan(v_mri)],
             weights=np.zeros_like(v_mri[~np.isnan(v_mri)]) + 1. / (v_mri[~np.isnan(v_mri)]).size,
             bins=logbins,
             histtype='bar',
             color='orangered',
             alpha=0.5,
             label='mri')
    plt.hist(v_lab[~np.isnan(v_lab)],
             weights=np.zeros_like(v_lab[~np.isnan(v_lab)]) + 1. / (v_lab[~np.isnan(v_lab)]).size,
             bins=logbins,
             histtype='bar',
             color='darkslategrey',
             alpha=0.5,
             label='lab')
    plt.legend(loc='upper right')
    plt.xscale('log')
    plt.savefig(op.join('img', 'velhist.svg'),
                transparent=True,
                bbox_inches="tight")


def S2SRMS():
    """
    A function to compute sample-to-sample RMS following the approach in
    Hooge et al., 2017
    (https://link.springer.com/content/pdf/10.3758/s13428-017-0955-x.pdf),
    as requested by reviewer 2.

    The idea behind the approach is to chunk the entire signal into windows and
    to compute sample to sample root mean squared distances from all the windows.
    Procedurally, in each window S2S-RMS are computed as the root average of
    squared distances between consecutive samples. After the window-wise computation,
    the median S2S-RMS within each "trial" (here: 15min run) is taken, and all
    median values are averaged. Windows in our context were 1000ms/1000 samples.

    The code below provides an implementation of this request, and the results.
    As we lay out in our response to the editor and the reviewer, though, we
    believe that the method reviewer 2 suggests, a precision estimation based on
    sample-to-sample distances in the entire signal, is a misguided method for
    professionally produced Hollywood movies. The substantial gaze control
    exerted by professionally produced movies (lightning, editing, deliberate
    composition; Dorr et al., 2010), and the limited spatial spread of gazes
    due to the fact that Hollywood movies center the viewers gaze in the middle
    of the screen (Goldstein et al., 2007) creates spatial closeness between
    successive samples that is artificially elevated. The resulting precision
    value underestimates the true noise, and misleads and distracts from the
    other noise estimations we have provided: Raw gaze samples, velocity distributions,
    percent signal loss, and references/citations of precision estimates that
    are based on 13 point calibration task validations under both MRI and lab
    conditions (i.e., precision estimates from dedicated precision assessments,
    not from the eyetracking signal acquired during moving watching).

    The results of this computation are included below. It should be obvious that
    the resulting numerical precision estimates seem implausible, especially
    given the obviously more unstable eyetracking signal plotted in the publication
    and the about 10 times higher precision estimates (i.e., 10 times worse)
    computed in the original studyforrest data publication from 13 point
    calibration procedures:

    OUTPUT:
    Currently processing data from lab sample
    \newcommand{\RMSlab}{0.05849002328619968}
    Currently processing data from mri sample
    \newcommand{\RMSmri}{0.061214090261903865}

    References:
    Dorr, M., Martinetz, T., Gegenfurtner, K. R., & Barth, E. (2010).
        Variability of eye movements when viewing dynamic natural scenes.
        Journal of vision , 10 (10).
        https://doi.org/10.1167/10.10.28
    Goldstein, R. B., Woods, R. L., & Peli, E. (2007).
        Where people look when watching movies: Do all viewers look at the same
        place? 37 (7), Computers in biology and medicine, 957-964.
        10.1016/j.compbiomed.2006.08.018
    Hanke, M., Adelhöfer, N., Kottke, D. et al. (2016)
        A studyforrest extension, simultaneous fMRI and eye gaze recordings
        during prolonged natural stimulation. Sci Data 3, 160092.
        https://doi.org/10.1038/sdata.2016.92

    """
    from scipy.spatial import distance

    # globs for all data for MRI and Lab
    datapath_mri = op.join('data', 'raw_eyegaze', 'sub-*', 'ses-movie', 'func',
                           'sub-*_ses-movie_task-movie_run-*_recording-eyegaze_physio.tsv.gz')
    datapath_lab = op.join('data', 'raw_eyegaze', 'sub-*', 'beh',
                           'sub-*_task-movie_run-*_recording-eyegaze_physio.tsv.gz')

    # compute results for both groups sequentially, first lab, then MRI
    for (data, assoc) in [(datapath_lab, 'lab'),
                          (datapath_mri, 'mri')]:
        infiles = glob(data)
        # make sure we have 15 subjects' data
        assert len(infiles) == 120
        print("Currently processing data from {} sample".format(assoc))
        # set px2deg for conversion to visual angles and window size (chose 1000
        # as a middle ground between "still computes withing a few hours" and
        # "create a lot of windows" (about 900 per 15 min segment).
        px2deg = 0.0266711972026 if assoc == 'lab' else 0.0185581232561
        window = 1000
        median_distances = []
        for f in infiles:
            data = np.genfromtxt(f,
                                  delimiter='\t',
                                  usecols=(0, 1) # get only x and y column
                                  )
            # chunk into windows
            datachunks = np.array_split(data, window)
            distances = []
            for chunk in datachunks:
                # calculate window-wise S2S-RSM, based on euclidean distance between
                # consecutive x-y-coordinates, and convert them into visual angles
                dist = distance.cdist(chunk, chunk, 'euclidean').diagonal(1) * px2deg
                # square the distances, average them, and square-root them
                RMS = np.sqrt(np.nanmean(np.square(dist)))
                distances.append(RMS)
            # calculate median RMS per run (~15 minute recording, i.e., median over
            # about 900 RMS values)
            median = np.median(distances)
            median_distances.append(median)
        # average the resulting 8 median RMSs
        meanRMS = np.nanmean(median_distances)
        # print results as Latex command using 'assoc' as sample identifier in name
        label_RMS = 'RMS{}'.format(assoc)
        print('\\newcommand{\\%s}{%s}'
              % (label_RMS, meanRMS))
        # save the results in variables
        if assoc == 'lab':
            RMS_lab = meanRMS
        elif assoc == 'mri':
            RMS_mri = meanRMS


def flowchart_figs():
    """
    Just for future reference: This is the subset of preprocessed and raw data
    used for the flowchart of the algorithm. Not to be executed.
    """
    import matplotlib.pyplot as plt
    from scipy import signal

    datapath = op.join('data', 'raw_eyegaze', 'sub-32', 'beh',
                        'sub-32_task-movie_run-1_recording-eyegaze_physio.tsv.gz')
    data = np.recfromcsv(datapath,
                         delimiter='\t',
                         names=['x', 'y', 'pupil', 'frame'])

    clf = EyegazeClassifier(
        px2deg=0.0266711972026,
        sampling_rate=1000.0)

    velocities = cal_velocities(data=data, sr=1000, px2deg=0.0266711972026)
    vel_subset_unfiltered = velocities[15200:17500]

    p = clf.preproc(data)
    # this is to illustrate PTn estimation and chunking
    vel_subset = p['vel'][15200:17500]
    fig, ax1 = plt.subplots()
    fig.set_figheight(2)
    fig.set_figwidth(7)
    fig.set_dpi(120)
    ax1.plot(
        vel_subset,
        color='black', lw=0.5)
    plt.close()

    # this is to illustrate preprocessing
    fig, ax1 = plt.subplots()
    fig.set_figheight(2)
    fig.set_figwidth(7)
    fig.set_dpi(120)
    ax1.plot(
        vel_subset,
        color='black', lw=0.5)
    ax1.plot(
        vel_subset_unfiltered,
        color='darkorange', ls='dotted', lw=0.5)
    plt.close()

    def _butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(
            order,
            normal_cutoff,
            btype='low',
            analog=False)
        return b,

    # Here is fixation and pursuit detection on Butterworth filtered subsets
    lp_cutoff_freq = 4.0
    sr=1000
    # let's get a data sample with no saccade
    win_data = p[16600:17000]
    b, a = _butter_lowpass(lp_cutoff_freq, sr)
    win_data['x'] = signal.filtfilt(b, a, win_data['x'], method='gust')
    win_data['y'] = signal.filtfilt(b, a, win_data['y'], method='gust')

    filtered_vels = cal_velocities(data=win_data, sr=1000, px2deg=0.0266711972026)
    fig, ax1 = plt.subplots()
    fig.set_figheight(2)
    fig.set_figwidth(7)
    fig.set_dpi(120)
    ax1.plot(
        filtered_vels,
        color='black', lw=0.5)
    plt.close()

def cal_velocities(data, sr, px2deg):
    """Helper to calculate velocities
    sr: sampling rate
    px2deg: conversion factor from pixel to degree
    """
    velocities = (np.diff(data['x']) ** 2 + np.diff(
        data['y']) ** 2) ** 0.5
    velocities *= px2deg * sr
    return velocities


def plot_raw_vel_trace():
    """
    Small helper function to plot raw velocity traces, as requested by reviewer 2
    in the second revision.
    """
    import matplotlib.pyplot as plt
    # use the same data as in savegaze() (no need for file retrieval, should be there)
    infiles = [
        op.join(
            'data',
            'raw_eyegaze', 'sub-32', 'beh',
            'sub-32_task-movie_run-5_recording-eyegaze_physio.tsv.gz'),
        op.join(
            'data',
            'raw_eyegaze', 'sub-02', 'ses-movie',  'func',
            'sub-02_ses-movie_task-movie_run-5_recording-eyegaze_physio.tsv.gz'
        ),
    ]
    # we need the sampling rate for plotting in seconds and velocity calculation
    sr = 1000
    # load data
    for i, f in enumerate(infiles):
        # read data
        data = np.recfromcsv(f,
                             delimiter='\t',
                             names=['x', 'y', 'pupil', 'frame'])

        # subset data. Hessels et al., 2017 display different noise levels on 4
        # second time series (ref. Fig 10). That still looks a bit dense, so we
        # go with 2 seconds, from start of 10sec excerpt to make it easier to
        # associate the 2 sec excerpt in to its place in the 10 sec excerpt
        # above
        data_subset = data[15000:17000]
        px2deg, ext = (0.0266711972026, 'lab') if '32' in f \
            else (0.0185581232561, 'mri')
        # take raw data and convert it to velocity: euclidean distance between
        # successive coordinate samples. Note: no entry for first datapoint!
        # Will plot all but first data point in other time series
        velocities = cal_velocities(data_subset, sr, px2deg)
        vel_color = 'xkcd:gunmetal'
        # prepare plotting - much manual setup, quite ugly - sorry
        fig, ax1 = plt.subplots()
        fig.set_figheight(2)
        fig.set_figwidth(7)
        fig.set_dpi(120)
        time_idx = np.linspace(0, len(data_subset) / sr, len(data_subset))[1:]
        max_x = float(len(data_subset) / sr)
        ax1.set_xlim(0, max_x)
        ax1.set_xlabel('time (seconds)')
        ax1.set_ylabel('coordinates')
        # left y axis set to max screensize in px
        ax1.set_ylim(0, 1280)
        # plot gaze trajectories (not preprocessed)
        ax1.plot(time_idx,
            data_subset['x'][1:],
            color='black', lw=1)
        ax1.plot(
            time_idx,
            data_subset['y'][1:],
            color='black', lw=1)
        # right y axis shows velocity "as is" (not preprocessed)
        ax2 = ax1.twinx()
        ax2.set_ylabel('velocity (deg/sec)', color=vel_color)
        ax2.tick_params(axis='y', labelcolor=vel_color)
        #ax2.set_yscale('log') ## TODO: Log scale or not?
        ax2.set_ylim(1, 2000)
        ax2.plot(time_idx,
            velocities,
            color=vel_color, lw=1)
        pl.savefig(
            op.join('img', 'rawtrace_{}.svg'.format(ext)),
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
            'data',
            'raw_eyegaze', 'sub-32', 'beh',
            'sub-32_task-movie_run-5_recording-eyegaze_physio.tsv.gz'),
        op.join(
            'data',
            'raw_eyegaze', 'sub-02', 'ses-movie',  'func',
            'sub-02_ses-movie_task-movie_run-5_recording-eyegaze_physio.tsv.gz'
        ),
    ]
    # one call per file due to https://github.com/datalad/datalad/issues/3356
    for f in infiles:
        dl.get(f)
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
        events = clf(p[15000:25000])
        # we remove plotting of details in favor of plotting raw gaze and
        # velocity traces with plot_raw_vel_trace() as requested by reviewer 2
        # in the second round of revision
        #events_detail = clf(p[24500:24750])

        fig = pl.figure(
            # fake size to get the font size down in relation
            figsize=(14, 2),
            dpi=120,
            frameon=False)
        ut.show_gaze(
            pp=p[15000:25000],
            events=events,
            sampling_rate=1000.0,
            show_vels=True,
            coord_lim=(0, 1280),
            vel_lim=(0, 1000))
        pl.savefig(
            op.join('img', 'remodnav_{}.svg'.format(ext)),
            transparent=True,
            bbox_inches="tight")
        pl.close()
        # plot details
        fig = pl.figure(
            # fake size to get the font size down in relation
            figsize=(7, 2),
            dpi=120,
            frameon=False)
        #ut.show_gaze(
        #    pp=p[24500:24750],
        #    events=events_detail,
        #    sampling_rate=1000.0,
        #    show_vels=True,
        #    coord_lim=(0, 1280),
        #    vel_lim=(0, 1000))
        #pl.savefig(
        #    op.join('img', 'remodnav_{}_detail.svg'.format(ext)),
        #    transparent=True,
        #    bbox_inches="tight")
        #pl.close()


def mainseq(s_mri,
            s_lab):
    """
    plot main sequences from movie data for lab and mri subjects.
    """
    import pandas as pd
    from matplotlib.lines import Line2D

    datapath = op.join('data',
                       'studyforrest-data-eyemovementlabels',
                       'sub*',
                       # limit to a single run, otherwise the resulting
                       # figure becomes so complex that it needs >16GB
                       # RAM to turn into an image for the manuscript,
                       # while the visible content hardly changes
                       '*run-2*.tsv')
    data = sorted(glob(datapath))
    from datalad.api import get
    get(dataset='.', path=data)

    # create dataframes for mri and lab subjects to plot in separate plots
    for (ids, select_sub, ext) in [
            (mri_ids, s_mri, 'mri'),
            (lab_ids, s_lab, 'lab')]:

        # load data from any file matching any of the subject IDs
        dfs = [
            pd.read_csv(f, header=0, delim_whitespace=True)
            for f in data
            if any('sub-{}'.format(i) in f for i in ids)
        ]
        df = pd.concat(dfs)

        # also create a dataframe for an individual subjects run
        sub = op.join('data',
                      'studyforrest-data-eyemovementlabels',
                      select_sub,
                      '{}_task-movie_run-2_events.tsv'.format(select_sub))
        sub_df = pd.read_csv(sub, header=0, delim_whitespace=True)

        for d, label in (
                (df, ''),
                (sub_df, '_sub')):
            # extract relevant event types
            SACCs = d[(d.label == 'SACC') | (d.label == 'ISAC')]
            PSOs = d[(d.label == 'HPSO') | (d.label == 'IHPS') | (d.label == 'LPSO') | (d.label == 'ILPS')]

            fig = pl.figure(
                # fake size to get the font size down in relation
                figsize=(6, 4),
                dpi=120,
                frameon=False)

            for ev, sym, color in (
                    (SACCs, '.', 'red'),
                    (PSOs, '+', 'darkblue'),
                    ):
                pl.loglog(
                    ev['amp'],
                    ev['peak_vel'],
                    sym,
                    # scale alpha down with increasing number of data points
                    alpha=min(0.1, 1.0 / max(0.0001, 0.002 * len(ev))),
                    color=color,
                    lw = 1,
                    rasterized=True
                )

            # cheat: custom legend to not propagate alpha into legend markers
            custom_legend = [
                Line2D([0], [0],
                       marker='.',
                       color='w',
                       markerfacecolor='red',
                       label='Saccade',
                       markersize=10),
                Line2D([0], [0],
                       marker='P',
                       color='w',
                       markerfacecolor='darkblue',
                       label='PSO',
                       #label='Low velocity PSOs',
                       markersize=10),
            ]

            pl.ylim((10.0, 1000))
            pl.xlim((0.01, 40.0))
            pl.legend(handles=custom_legend, loc=4)
            pl.ylabel('peak velocities (deg/s)')
            pl.xlabel('amplitude (deg)')
            pl.savefig(
                op.join(
                    'img',
                    'mainseq{}_{}.svg'.format(
                        label,
                        ext)),
                transparent=True,
                bbox_inches="tight")
            pl.close()


def RMSD(mn,
         sd,
         no):
    """
    Compute our interpretation of the RMSD, following equation 2 in
    Andersson et al., 2017

    Parameters
    ----------
    mn, sd, no
      lists with mean, standard deviation, and number of events for
      a given event type and stimulus type for all available algorithms.

    Returns
    -------
    1d-array
      RMSD score per "algorithm". The first two values represent the
      human coders."""

    per_param = []
    # convert params to array
    mn, sd, no = np.array(mn), np.array(sd), np.array(no)
    # compute the root mean square difference between algorithm and mean
    # of coders per parameter and algorithm
    for l in [mn, sd, no]:
        l_scaled = l / float(np.max(l))
        l_alg = np.sqrt((
            # all scores
            # minus the average of the humans
            l_scaled - np.mean(l_scaled[:2])) ** 2
        )
        per_param.append(l_alg)
    # sum the root mean square differences per algorithm across parameters
    # also give the human rater performance as the first two values
    # argsorting twice gets us the ranks
    return np.array(per_param).sum(axis=0).argsort().argsort()


def get_remodnav_params(stim_type):
    """
    Function to generate distribution parameters for event types.
    Used for the RMSD computation.

    Parameters
    ----------
    stim_type = one str of 'img', 'dots', 'video'

    Returns
    -------
      a dictionary with distribution parameters for all events for the given
      stim_type
    """
    # iterate through stim_types
    events = []
    # the data files exist twice (one per coder). The raw eye gaze data in corresponding
    # files should be the same, so I assume its safe to just take one coders files.
    src = 'RA'
    for fname in labeled_files[stim_type]:
        data, target_labels, target_events, px2deg, sr = \
            load_anderson(stim_type, fname.format(src))

        clf = EyegazeClassifier(
            px2deg=px2deg,
            sampling_rate=sr,
        )
        p = clf.preproc(data)
        events.extend(clf(p))
    for ev in events:
        ev['label'] = label_map[ev['label']]

    from collections import OrderedDict
    durs = OrderedDict()
    durs['event']=[]
    durs['alg']=[]
    durs['mn']=[]
    durs['sd']=[]
    durs['no']=[]
    # iterate through relabeled event types
    for ev_type in ['FIX', 'SAC', 'PUR', 'PSO']:
        durations = get_durations(events, ev_type)
        durs['event'].append(ev_type)
        durs['mn'].append(int(np.nanmean(durations) * 1000))
        durs['sd'].append(int(np.nanstd(durations) * 1000))
        durs['no'].append(len(durations))
        durs['alg'].append('RE')

    return durs


def print_RMSD():
    """
    Function to generate tables 3, 4, 5, partial 6 from Andersson et al., 2017
    for use in main.tex.
    """
    # I don't want to overwrite the original dicts
    from copy import deepcopy
    img = deepcopy(image_params)
    dots = deepcopy(dots_params)
    video = deepcopy(video_params)
    event_types = ['FIX', 'SAC', 'PSO', 'PUR']

    for stim in ['img', 'dots', 'video']:
        durs = get_remodnav_params(stim)
        dic = [img if stim == 'img' else dots if stim == 'dots' else video]
        # append the parameters produced by remodnav to the other algorithms'
        for ev in event_types:
            for p in ['mn', 'sd', 'no', 'alg']:
                # unfortunately, dic is a list now...thats why [0] is there.
                # index the dicts with the position of the respective event type
                dic[0][ev][p].append(durs[p][durs['event'].index(ev)])
            # print results as LaTeX commands
            # within a stim_type, we iterate over keys (events and params) in the nested dicts
            for par in ['mn', 'sd', 'no']:
                # index the values of the dist params in the nested dicts with the position
                # of the respective algorithm.
                for alg in dic[0][ev]['alg']:
                    label_prefix = '{}{}{}{}'.format(ev, stim, par, alg)
                    # take the value of the event and param type by indexing the dict with the position of
                    # the current algorithm
                    print('\\newcommand{\\%s}{%s}'
                          %(label_prefix, dic[0][ev][par][dic[0][ev]['alg'].index(alg)]))
        # compute RMSDs for every stimulus category
        for ev in event_types:
            rmsd = RMSD(dic[0][ev]['mn'],
                        dic[0][ev]['sd'],
                        dic[0][ev]['no'])
            # print results as LaTeX commands
            algo = dic[0][ev]['alg']
            for i in range(len(rmsd)):
                label = 'rank{}{}{}'.format(ev, stim, algo[i])
                print('\\newcommand{\\%s}{%s}'
                      %(label, rmsd[i]))


def plot_dist(figures):
    """
    Plot the events duration distribution per movie run, per data set.
    """
    import pandas as pd

    # do nothing if we don't want to plot
    if not figures:
        return

    import datalad.api as dl
    dl.install(op.join('data', 'studyforrest-data-eyemovementlabels'))
    datapath = op.join('data',
                       'studyforrest-data-eyemovementlabels',
                       'sub*',
                       '*.tsv')

    data = sorted(glob(datapath))
    dl.get(dataset='.', path=data)

    for ds, ds_name in [(mri_ids, 'mri'), (lab_ids, 'lab')]:
        dfs = [
            pd.read_csv(f, header=0, delim_whitespace=True)
            for f in data
            if any('sub-{}'.format(i) in f for i in ds)
        ]
        df = pd.concat(dfs)
        # thats a concatinated dataframe with all files from one dataset (lab or mri)
        # extract relevant event types
        SACs = df[(df.label == 'SACC') | (df.label == 'ISACS')]
        FIX = df[df.label == 'FIXA']
        PSOs = df[(df.label == 'HPSO') | (df.label == 'IHPS') | (df.label == 'LPSO') | (df.label == 'ILPS')]
        PURs = df[df.label == 'PURS']
        # plot a histogram. Set the same x-axis limits as NH for fixations and saccades,
        # and exclude outlying 0.5% for other events
        for (ev_df, label, x_lim, y_lim) in [
                (SACs, 'saccade', (0, 0.16), (1, 62000)),
                (FIX, 'fixation', (0, 1.0), (1, 50000)),
                (PSOs, 'PSO', (0, 0.04), (1, 26000)),
                (PURs, 'pursuit', (0, .8), (1, 30000))]:
            fig = pl.figure(figsize=(3,2))
            pl.hist(ev_df['duration'].values,
                    bins='doane',
                    range=x_lim,
                    color='gray')
                    #log=True)
            pl.xlabel('{} duration in s'.format(label))
            pl.xlim(x_lim)
            pl.ylim(y_lim)
            pl.savefig(
                op.join(
                    'img',
                    'hist_{}_{}.svg'.format(
                        label,
                        ds_name)),
                transparent=True,
                bbox_inches="tight")
            pl.close()


def kappa():
    """
    During the review process, reviewer 2 requested Cohens Kappa computation.
    We have not implemented the measure before because we felt it did not add
    information beyond the confusion and RMSD computations.
    """
    px2deg = None
    sr = None
    from sklearn.metrics import cohen_kappa_score
    # for every stimulus type
    for stim in ['img', 'dots', 'video']:
        # for every eye movement label used in Anderson et al. (2017)
        for (ev, i) in [('Fix', 1), ('Sac', 2), ('PSO', 3)]:
            # initialize lists to store classification results in
            RA_res = []
            MN_res = []
            AL_res = []
            # aggregate the target_labels of all files per coder + stim_type
            for idx, fname in enumerate(labeled_files[stim]):
                for coder in ['MN', 'RA', 'AL']:
                    if coder in ['MN', 'RA']:
                        data, target_labels, target_events, px2deg, sr = \
                            load_anderson(stim, fname.format(coder))
                        # dichotomize classification based on event type
                        labels = [1 if j == i else 0 for j in target_labels]
                        if coder == 'MN':
                            MN_res.append(labels)
                        elif coder == 'RA':
                            RA_res.append(labels)
                    else:
                        # get REMoDNaV classification
                        clf = EyegazeClassifier(
                            px2deg=px2deg,
                            sampling_rate=sr,
                        )
                        p = clf.preproc(data)
                        events = clf(p)

                        # convert event list into anderson-style label array
                        l = np.zeros(target_labels.shape, target_labels.dtype)
                        for e in events:
                            l[int(e['start_time'] * sr):int((e['end_time']) * sr)] = \
                                anderson_remap[label_map[e['label']]]
                        # dichotomize REMoDNaV classification results as well
                        labels = [1 if j == i else 0 for j in l]
                        AL_res.append(labels)

                if len(MN_res[idx]) != len(RA_res[idx]):
                    print(
                        "% #\n% # %INCONSISTENCY Found label length mismatch "
                        "between coders for: {}\n% #\n".format(fname))
                    shorter = min([len(RA_res[idx]), len(MN_res[idx])])
                    print('% Truncate labels to shorter sample: {}'.format(
                        shorter))
                    # truncate the labels by indexing up to the highest index
                    # in the shorter list of labels
                    MN_res[idx] = MN_res[idx][:shorter]
                    RA_res[idx] = RA_res[idx][:shorter]
                    AL_res[idx] = AL_res[idx][:shorter]
            # dummy check whether we really have the same number of files per coder
            assert len(RA_res) == len(MN_res)
            # flatten the list of lists
            RA_res_flat = [item for sublist in RA_res for item in sublist]
            MN_res_flat = [item for sublist in MN_res for item in sublist]
            AL_res_flat = [item for sublist in AL_res for item in sublist]
            #print(sum(RA_res_flat), sum(MN_res_flat))
            assert len(RA_res_flat) == len(MN_res_flat) == len(AL_res_flat)
            # compute Cohens Kappa
            for rating, comb in [('RAMN', [RA_res_flat, MN_res_flat]),
                                 ('ALRA', [RA_res_flat, AL_res_flat]),
                                 ('ALMN', [MN_res_flat, AL_res_flat])]:
                kappa = cohen_kappa_score(comb[0], comb[1])
                label = 'kappa{}{}{}'.format(rating, stim, ev)
                print('\\newcommand{\\%s}{%s}' % (label, '%.2f' % kappa))


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
    parser.add_argument(
        '-m', '--mainseq',
        help='if given, mainsequence plots are produced.',
        action='store_true', default=False)
    parser.add_argument(
        '--sublab',
        help='individual lab-subject for single main sequence',
        default='sub-27')
    parser.add_argument(
        '--submri',
        help='individual mri-subject for single main sequence',
        default='sub-17')


    args = parser.parse_args()
    # generate & save figures; export the stats
    if args.figure or args.stats:
        savefigs(args.figure, args.stats)
        print_RMSD()
        plot_dist(args.figure)
        kappa()
        plot_raw_vel_trace()
    if args.mainseq:
        mainseq(args.submri, args.sublab)
    if args.remodnav:
        savegaze()
