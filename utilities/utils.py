import numpy as np
import scipy


def model_responsive_and_selective_tests(output, sessions, t, width=0.5, base_left=-1.0, correct_only=True):
    
    res = {}
    
    for session in sessions:
        
        # trial labels and data
        y_stim = output['y_stims'][session]
        y_left = output['y_lefts'][session]
        y_outcome = output['y_outcomes'][session]
        X = output['Xs'][session] # trials x neurons x time
        if correct_only:
            trial_stim = y_stim[y_outcome == 1]
            trial_choice = y_left[y_outcome == 1]
            X = X[y_outcome == 1, :, :]
        else:
            trial_stim = y_stim
            trial_choice = y_left
        n_trials, n_neurons, n_time = X.shape
            
        # time windows
        t_baseline = (t >= base_left) & (t < (base_left + width))
        t_sampling = (t >= 0) & (t < width)
        t_D = t[-1] - 1 + 0.025
        t_delay = (t >= t_D - width) & (t < t_D)
            
        # calculate firing rates
        res_sub = {}
        for i_neuron in range(n_neurons):
            fr_baseline = np.full(n_trials, np.nan)
            fr_sampling = np.full(n_trials, np.nan)
            fr_delay = np.full(n_trials, np.nan)
            for i_trial in range(n_trials):
                fr_baseline[i_trial] = X[i_trial, i_neuron, t_baseline].mean()
                fr_sampling[i_trial] = X[i_trial, i_neuron, t_sampling].mean()
                fr_delay[i_trial] = X[i_trial, i_neuron, t_delay].mean()
            # statistical tests
            res_responsive_sampling = scipy.stats.mannwhitneyu(fr_baseline, fr_sampling)
            res_responsive_delay = scipy.stats.mannwhitneyu(fr_baseline, fr_delay)
            fr_sampling_1, fr_sampling_2 = fr_sampling[trial_stim > 50], fr_sampling[trial_stim < 50]
            fr_delay_1, fr_delay_2 = fr_delay[trial_choice == 1], fr_delay[trial_choice == 0]
            res_selective_sampling = scipy.stats.mannwhitneyu(fr_sampling_1, fr_sampling_2)
            res_selective_delay = scipy.stats.mannwhitneyu(fr_delay_1, fr_delay_2)
            res_sub[i_neuron] = {
                'p_responsive_sampling': res_responsive_sampling.pvalue,
                'p_responsive_delay': res_responsive_delay.pvalue,
                'p_selective_sampling': res_selective_sampling.pvalue,
                'p_selective_delay': res_selective_delay.pvalue,
                'fr_baseline': fr_baseline,
                'fr_sampling': fr_sampling,
                'fr_sampling_1': fr_sampling_1,
                'fr_sampling_2': fr_sampling_2,
                'fr_delay': fr_delay,
                'fr_delay_1': fr_delay_1,
                'fr_delay_2': fr_delay_2,
                'psth': X[:, i_neuron, :].mean(axis=0),
            }
            
        res[session] = res_sub
    
    return res


def responsive_and_selective_tests(data, sessions, 
                                   width=0.5, correct_only=True, t_base=2.5, t_base_rel_to='T', d_base=5.0, d_base_rel_to='D',
                                   bin_size=None):
    
    res = {}
    
    for session in sessions:
        
        # trial labels
        trial_stim_correct = data['data_all'][0, session][7]
        trial_stim_error = data['data_all'][0, session][8]
        if correct_only:
            trial_stim = trial_stim_correct.flatten()
            trial_choice = (trial_stim_correct > 50).astype(float).flatten()
        else:
            trial_stim = np.vstack((trial_stim_correct, trial_stim_error)).flatten()
            trial_choice = np.vstack(((trial_stim_correct > 50).astype(float), (trial_stim_error < 50).astype(float))).flatten()
            
        # spike trains
        spike_trains = data['data_all'][0, session][-1]

        # behavior timestamps
        central_licks_correct = data['data_all'][0, session][9] + data['data_all'][0, session][11]
        central_licks_error = data['data_all'][0, session][10] + data['data_all'][0, session][12]
        if correct_only:
            central_licks = central_licks_correct.flatten()
        else:
            central_licks = np.vstack((central_licks_correct, central_licks_error)).flatten()
        IEIs_correct = data['data_all'][0, session][15]
        IEIs_error = data['data_all'][0, session][16] 
        if correct_only:
            lateral_licks = central_licks + IEIs_correct.flatten()
        else:
            lateral_licks = central_licks + np.vstack((IEIs_correct, IEIs_error)).flatten()
            
        # calculate firing rates
        res_sub = {}
        if bin_size is not None:
            t_sampling_L, t_sampling_R = 0.5, 1.0
            n_sampling_left = int(np.ceil(t_sampling_L / bin_size))
            n_sampling_right = int(np.ceil(t_sampling_R / bin_size))
            t_sampling = [-t_sampling_L + i * bin_size for i in range(n_sampling_left)] + [i * bin_size for i in range(n_sampling_right)]
            t_sampling = np.array(t_sampling) + bin_size / 2
            t_delay_L, t_delay_R = 1.0, 0.5
            n_delay_left = int(np.ceil(t_delay_L / bin_size))
            n_delay_right = int(np.ceil(t_delay_R / bin_size))
            t_delay = [-t_delay_L + i * bin_size for i in range(n_delay_left)] + [i * bin_size for i in range(n_delay_right)]
            t_delay = np.array(t_delay) + bin_size / 2
        for i_neuron in range(len(spike_trains)):
            spike_train = spike_trains[i_neuron][0][0]
            fr_sampling = np.full(len(central_licks), np.nan)
            fr_sampling_base = np.full(len(central_licks), np.nan)
            fr_delay = np.full(len(lateral_licks), np.nan)
            fr_delay_base = np.full(len(lateral_licks), np.nan)
            if bin_size is not None:
                X_sampling = []
                X_delay = []
            for i_trial in range(len(central_licks)):
                train_T = spike_train.copy() - central_licks[i_trial]
                train_D = spike_train.copy() - lateral_licks[i_trial]
                fr_sampling[i_trial] = ((train_T >= 0) & (train_T < width)).sum() / width
                if t_base_rel_to == 'T':
                    fr_sampling_base[i_trial] = ((train_T >= -(t_base + width)) & (train_T < -t_base)).sum() / width
                elif t_base_rel_to == 'D':
                    fr_sampling_base[i_trial] = ((train_D >= -(t_base + width)) & (train_D < -t_base)).sum() / width
                fr_delay[i_trial] = ((train_D >= -width) & (train_D < 0)).sum() / width
                if d_base_rel_to == 'T':
                    fr_delay_base[i_trial] = ((train_T >= -(d_base + width)) & (train_T < -d_base)).sum() / width
                elif d_base_rel_to == 'D':
                    fr_delay_base[i_trial] = ((train_D >= -(d_base + width)) & (train_D < -d_base)).sum() / width
                if bin_size is not None:
                    X_sampling.append([
                        ((train_T >= tp - bin_size / 2) & (train_T < tp + bin_size / 2)).sum() / bin_size for tp in t_sampling
                    ])
                    X_delay.append([
                        ((train_D >= tp - bin_size / 2) & (train_D < tp + bin_size / 2)).sum() / bin_size for tp in t_delay
                    ])
            if bin_size is None:
                psth_sampling, t_sampling = None, None
                psth_delay, t_delay = None, None
            else:
                psth_sampling = np.array(X_sampling).mean(axis=0)
                psth_delay = np.array(X_delay).mean(axis=0)
            # statistical tests
            res_responsive_sampling = scipy.stats.mannwhitneyu(fr_sampling_base, fr_sampling)
            res_responsive_delay = scipy.stats.mannwhitneyu(fr_delay_base, fr_delay)
            res_selective_sampling = scipy.stats.mannwhitneyu(fr_sampling[trial_stim > 50], fr_sampling[trial_stim < 50])
            res_selective_delay = scipy.stats.mannwhitneyu(fr_delay[trial_choice == 1], fr_delay[trial_choice == 0])
            res_sub[i_neuron] = {
                'p_responsive_sampling': res_responsive_sampling.pvalue,
                'p_responsive_delay': res_responsive_delay.pvalue,
                'p_selective_sampling': res_selective_sampling.pvalue,
                'p_selective_delay': res_selective_delay.pvalue,
                'fr_sampling': fr_sampling,
                'fr_sampling_base': fr_sampling_base,
                'fr_delay': fr_delay,
                'fr_delay_base': fr_delay_base,
                't_sampling': t_sampling,
                'psth_sampling': psth_sampling,
                't_delay': t_delay,
                'psth_delay': psth_delay
            }
            
        res[session] = res_sub
    
    return res


def get_warped_data(data, session, bin_size, padding, sessions=None, mean_correct_only=False):
    
    # get trial labels
    trial_stim_correct = data['data_all'][0, session][7]
    trial_stim_error = data['data_all'][0, session][8]
    trial_stim = np.vstack((trial_stim_correct, trial_stim_error)).flatten()
    trial_choice = np.vstack(((trial_stim_correct > 50).astype(float), (trial_stim_error < 50).astype(float))).flatten()
    trial_outcome = (((trial_stim > 50) & (trial_choice == 1)) | ((trial_stim < 50) & (trial_choice == 0))).astype(float)
    
    # construct warped time vector
    if sessions is None:
        n_sessions = data['data_all'].shape[1]
        sessions = range(n_sessions)
    else:
        n_sessions = len(sessions)
    if mean_correct_only:
        IEI_mean = np.sum([np.sum(data['data_all'][0, sess][15]) for sess in sessions]) / \
                           np.sum([len(data['data_all'][0, sess - 1][15]) for sess in sessions])
    else:
        IEI_mean = np.sum([np.sum(data['data_all'][0, sess - 1][15]) + \
                           np.sum(data['data_all'][0, sess - 1][16]) for sess in sessions]) / \
                   np.sum([len(data['data_all'][0, sess - 1][15]) + \
                           len(data['data_all'][0, sess - 1][16]) for sess in sessions])
    n_bins = round(IEI_mean / bin_size)
    n_padding_bins = round(padding / bin_size)
    t = np.squeeze(np.hstack((
        -bin_size * np.fliplr(np.reshape(np.arange(1, n_padding_bins + 1), (1, -1))) + bin_size / 2,
        bin_size * np.reshape(np.arange(n_bins), (1, -1)) + bin_size / 2,
        bin_size * np.reshape(np.arange(n_padding_bins), (1, -1)) + n_bins * bin_size + bin_size / 2
    )))
    
    # get the spike data
    spike_trains = data['data_all'][0, session][-1]
    
    # get the behavioral timestamps
    central_licks_correct = data['data_all'][0, session][9] + data['data_all'][0, session][11]
    central_licks_error = data['data_all'][0, session][10] + data['data_all'][0, session][12]
    central_licks = np.vstack((central_licks_correct, central_licks_error))
    IEIs_correct = data['data_all'][0, session][15]
    IEIs_error = data['data_all'][0, session][16]    
    IEIs = np.vstack((IEIs_correct, IEIs_error))
    
    # binning
    X = np.empty((central_licks.shape[0], len(spike_trains), len(t)))
    for j in range(len(spike_trains)):
        spike_train = spike_trains[j][0][0]
        for i in range(central_licks.shape[0]):
            train = spike_train - central_licks[i, 0]
            IEI = IEIs[i, 0]
            train = train[(train >= -padding) & (train < IEI + padding)]
            for b in range(n_padding_bins):
                X[i, j, b] = np.sum(((train >= -bin_size * (n_padding_bins - b)) & \
                    (train < -bin_size * (n_padding_bins - b - 1))).astype(float)) / bin_size
                X[i, j, n_padding_bins + n_bins + b] = np.sum(((train >= IEI + bin_size * b) & \
                    (train < IEI + bin_size * (b + 1))).astype(float)) / bin_size
            for b in range(n_bins):
                X[i, j, n_padding_bins + b] = np.sum(((train >= IEI / n_bins * b) & \
                    (train < IEI / n_bins * (b + 1))).astype(float)) / (IEI / n_bins)
        
    return (X, trial_stim, trial_choice, trial_outcome, t)


def raster_format(spikes, i):
    x = np.vstack([spikes.reshape(1, -1), spikes.reshape(1, -1)])
    y = np.vstack([
        np.array([i + 0.4] * len(spikes)).reshape(1, -1), 
        np.array([i - 0.4] * len(spikes)).reshape(1, -1)
    ])
    return (x, y)


def delta_auROC(r1, r2):
    '''
    comparison of distributions r1(t) and r2(t)
    
    inputs r1 and r2 must be 2-D arrays (trials x time) with matching time dimension
    
    output in [0, 1] with 0 indicating significantly higher r2 and 1 indicating significantly higher r1
    '''
    assert r1.shape[1] == r2.shape[1]
    dauROC = np.full(r1.shape[1], np.nan)
    for i_t in range(r1.shape[1]):
        r1_t = r1[:, i_t]
        r2_t = r2[:, i_t]
        x = []
        y = []
        thresh_min = int(np.floor(min(list(r1_t) + list(r2_t)))) - 1
        thresh_max = int(np.ceil(max(list(r1_t) + list(r2_t)))) + 1
        thresh_range = np.unique([thresh_min] + list(r1_t) + list(r2_t) + [thresh_max])
        # thresh_range = range(thresh_min, thresh_max + 1)
        for thresh in thresh_range:
            x.append( (r2_t > thresh).sum() / len(r2_t) )
            y.append( (r1_t > thresh).sum() / len(r1_t) )
        x = x[::-1]
        y = y[::-1]
        dauROC[i_t] = sum([ np.mean(y[i:(i + 2)]) * (x[i + 1] - x[i]) for i in range(len(x) - 1) ])
    return dauROC


def smooth(x, type_, width):
    if type_ is None or width == 1:
        return x
    if (width // 2) == (width / 2):
        width += 1 # force odd-number width
        print('Warning: kernel width changed to', width)
    if type_ == 'box':
        z = x.copy()
        for i in range(len(x)):
            mask = (np.arange(len(x)) >= i - (width - 1) / 2) & \
            (np.arange(len(x)) <= i + (width - 1) / 2)
            z[i] = np.mean(x[mask])
    elif type_ == 'gaussian':
        n = np.linspace(-(width - 1) / 2, (width - 1) / 2, num=width)
        gauss_win = np.exp(-0.5 * (n / ((width - 1) / 5)) ** 2) # translated from MATLAB's 'gausswin'
        gauss_win /= gauss_win.sum()
        z = np.convolve(x, gauss_win, mode='same')
    return z


def min_dist_classify(X, y, fun_dist):
    unique_labels = np.unique(y)
    n_unique_labels = len(unique_labels)
    n_per_label = np.array([np.sum(y == label) for label in unique_labels])
    n_bins = X.shape[-1]
    accuracy_over_time = np.zeros((n_unique_labels, n_bins))
    for i_bin in range(n_bins):
        X_t = X[:, :, i_bin]
        for i_leave_out in range(X_t.shape[0]):
            x_leave_out = X_t[i_leave_out, :]
            leave_out_mask = (np.arange(X_t.shape[0]) != i_leave_out)
            correct_label = y[i_leave_out]
            correct_label_ind = np.argmax(correct_label == unique_labels)
            centroids = [X_t[leave_out_mask & (y == label), :].mean(axis=0) for label in unique_labels]
            distances = [fun_dist(x_leave_out, x) for x in centroids]
            predict_label = unique_labels[np.argmax(distances == min(distances))]
            if (predict_label == correct_label):
                accuracy_over_time[correct_label_ind, i_bin] += 1
    accuracy_over_time /= n_per_label.reshape((-1, 1))
    return accuracy_over_time


def fit_shape_templates(x_data, y_data, options=None):

    # validate inputs
    x_data = np.array(x_data)
    if (len(x_data) == 0) or (len(x_data.shape) > 1):
        raise ValueError('x_data must be a non-empty list or 1D numpy array.')
    y_data = np.array(y_data)
    if (len(y_data) == 0) or (len(y_data.shape) > 1):
        raise ValueError('y_data must be a non-empty list or 1D numpy array.')
    if (len(x_data) != len(y_data)):
        raise ValueError('Mismatching x and y inputs.')
    if (options is not None):
        if (type(options) != list):
            raise ValueError("options must be a list of dictionaries with key 'shape'")
        if (type(options[0]) != dict):
            raise ValueError("options must be a list of dictionaries with key 'shape'")
        if any(['shape' not in dict_ for dict_ in options]):
            raise ValueError("options must be a list of dictionaries with key 'shape'")

    output = dict()
    output['x_data'] = x_data
    output['y_data'] = y_data

    all_shape_templates = ['linear', 'v', 'step']

    if (options is None):
        # default: all fits, unrestricted
        requested_shapes = all_shape_templates
    else:
        requested_shapes = [shape.lower() for shape in [dict_['shape'] for dict_ in options]]

    # --- linear (2-parameter fit) ---
    # parameters are:
    # slope
    # y_intercept
    if ('linear' in requested_shapes):
        fun_line = lambda p, x : p[0] * x + p[1]
        n_params = 2
        y_data_clean = y_data[~np.isnan(y_data)]
        if (len(y_data_clean) < (n_params + 1)) or all((y_data_clean - np.mean(y_data_clean)) == 0.0):
            # not enough data points to fit line...
            output['params_linear'] = []
            output['fit_linear'] = []
            output['fit_linear_full'] = []
            output['F_stat_linear'] = -np.inf
            output['p_linear'] = np.inf
        else:
            x_data_clean = x_data[~np.isnan(y_data)]
            if (options is None):
                # unrestricted fit
                params = np.polyfit(x_data_clean, y_data_clean, 1)
                min_y_range = 0
            else:
                # restricted fit
                idx = np.argmax([dict_['shape'] == 'linear' for dict_ in options])
                lb = (options[idx]['y_min'] if ('y_min' in options[idx]) else -np.inf)
                ub = (options[idx]['y_max'] if ('y_max' in options[idx]) else np.inf)
                min_y_range = (options[idx]['min_y_range'] if ('min_y_range' in options[idx]) else 0)
                fun_loss = lambda p : np.sum((fun_line(p, x_data_clean) - y_data_clean) ** 2)
                if np.isinf(lb):
                    if np.isinf(ub):
                        params = np.polyfit(x_data_clean, y_data_clean, 1)
                    else:
                        constraints = (
                            # y(0) <= ub
                            {'type': 'ineq', 'fun': lambda p: ub - p[1]},
                            # y(100) <= ub
                            {'type': 'ineq', 'fun': lambda p: ub - p[0] * 100 - p[1]}
                        )
                        res = scipy.optimize.minimize(fun_loss, [0, np.mean(y_data_clean)], constraints=constraints)
                        params = res.x
                else:
                    if np.isinf(ub):
                        constraints = (
                            # y(0) >= lb
                            {'type': 'ineq', 'fun': lambda p: p[1] - lb},
                            # y(100) >= lb
                            {'type': 'ineq', 'fun': lambda p: p[0] * 100 + p[1] - lb},
                        )
                        res = scipy.optimize.minimize(fun_loss, [0, np.mean(y_data_clean)], constraints=constraints)
                        params = res.x
                    else:
                        constraints = (
                            # y(0) >= lb
                            {'type': 'ineq', 'fun': lambda p: p[1] - lb},
                            # y(0) <= ub
                            {'type': 'ineq', 'fun': lambda p: ub - p[1]},
                            # y(100) >= lb
                            {'type': 'ineq', 'fun': lambda p: p[0] * 100 + p[1] - lb},
                            # y(100) <= ub
                            {'type': 'ineq', 'fun': lambda p: ub - p[0] * 100 - p[1]}
                        )
                        res = scipy.optimize.minimize(fun_loss, [0, np.mean(y_data_clean)], constraints=constraints)
                        params = res.x
            sse2 = np.sum((fun_line(params, x_data_clean) - y_data_clean) ** 2)
            if (sse2 == 0):
                print('Warning: linear fit to ( x =', x_data_clean, 'y =', y_data_clean, ') is perfect, will raise divide-by-0 warning')
            sse1 = np.sum((np.mean(y_data_clean) - y_data_clean) ** 2)
            df2 = len(y_data_clean) - n_params
            df1 = n_params - 1
            F_stat = df2 / df1 * (sse1 - sse2) / sse2
            p = scipy.stats.f.sf(F_stat, df1, df2)
            output['params_linear'] = params
            output['fit_linear'] = fun_line(params, x_data)
            output['fit_linear_full'] = (np.linspace(np.min(x_data), np.max(x_data)),
                                         fun_line(params, np.linspace(np.min(x_data), np.max(x_data))))
            output['F_stat_linear'] = F_stat
            output['p_linear'] = p
            y_range = np.max(fun_line(params, x_data)) - np.min(fun_line(params, x_data))
            if (y_range < min_y_range):
                output['F_stat_linear'] = -np.inf
                output['p_linear'] = np.inf

    # --- V (2-parameter fit --- a priori, knee of V occurs at 50) ---
    # parameters are:
    # height / 50
    # vertical_shift - height
    if ('v' in requested_shapes):
        fun_V = lambda p, x : np.concatenate([-p[0] * (x[x < 50] - 50) + p[1],
                                              p[0] * (x[x >= 50] - 50) + p[1]])
        y_data_clean = y_data[~np.isnan(y_data)]
        if (len(y_data_clean) < (n_params + 1)) or all((y_data_clean - np.mean(y_data_clean)) == 0.0):
            # not enough data points to fit V...
            output['params_v'] = []
            output['fit_v'] = []
            output['fit_v_full'] = []
            output['F_stat_v'] = -np.inf
            output['p_v'] = np.inf
        else:
            x_data_clean = x_data[~np.isnan(y_data)]
            # to fit V with midpoint at 50, fold left onto right and do a linear regression
            x_data_clean_fold = np.concatenate([50 - x_data_clean[x_data_clean < 50], 
                                                x_data_clean[x_data_clean >= 50] - 50])
            if (options is None):
                # unconstrained fit
                n_params = 2
                params = np.polyfit(x_data_clean_fold, y_data_clean, 1)
                min_y_range = 0
            else:
                # restricted fit
                idx = np.argmax([dict_['shape'] == 'v' for dict_ in options])
                lb = (options[idx]['y_min'] if ('y_min' in options[idx]) else -np.inf)
                ub = (options[idx]['y_max'] if ('y_max' in options[idx]) else np.inf)
                min_y_range = (options[idx]['min_y_range'] if ('min_y_range' in options[idx]) else 0)
                fun_line = lambda p, x : p[0] * x + p[1]
                fun_loss = lambda p : sum((fun_line(p, x_data_clean_fold) - y_data_clean) ** 2)
                if np.isinf(lb):
                    if np.isinf(ub):
                        params = np.polyfit(x_data_clean_fold, y_data_clean, 1)
                    else:
                        constraints = (
                            # y(0) <= ub
                            {'type': 'ineq', 'fun': lambda p: ub - p[1]},
                            # y(50) <= ub
                            {'type': 'ineq', 'fun': lambda p: ub - p[0] * 50 - p[1]}
                        )
                        res = scipy.optimize.minimize(fun_loss, [0, np.mean(y_data_clean)], constraints=constraints)
                        params = res.x
                else:
                    if np.isinf(ub):
                        constraints = (
                            # y(0) >= lb
                            {'type': 'ineq', 'fun': lambda p: p[1] - lb},
                            # y(50) >= lb
                            {'type': 'ineq', 'fun': lambda p: p[0] * 50 + p[1] - lb},
                        )
                        res = scipy.optimize.minimize(fun_loss, [0, np.mean(y_data_clean)], constraints=constraints)
                        params = res.x
                    else:
                        constraints = (
                            # y(0) >= lb
                            {'type': 'ineq', 'fun': lambda p: p[1] - lb},
                            # y(0) <= ub
                            {'type': 'ineq', 'fun': lambda p: ub - p[1]},
                            # y(50) >= lb
                            {'type': 'ineq', 'fun': lambda p: p[0] * 50 + p[1] - lb},
                            # y(50) <= ub
                            {'type': 'ineq', 'fun': lambda p: ub - p[0] * 50 - p[1]}
                        )
                        res = scipy.optimize.minimize(fun_loss, [0, np.mean(y_data_clean)], constraints=constraints)
                        params = res.x
            sse2 = np.sum((fun_V(params, x_data_clean) - y_data_clean) ** 2)
            if (sse2 == 0):
                print('Warning: V fit to ( x =', x_data_clean, 'y =', y_data_clean, ') is perfect, will raise divide-by-0 warning')
            sse1 = np.sum((np.mean(y_data_clean) - y_data_clean) ** 2)
            df2 = len(y_data_clean) - n_params
            df1 = n_params - 1
            F_stat = df2 / df1 * (sse1 - sse2) / sse2
            p = scipy.stats.f.sf(F_stat, df1, df2)
            output['params_v'] = params
            output['fit_v'] = fun_V(params, x_data)
            output['fit_v_full'] = (np.linspace(np.min(x_data), np.max(x_data)), 
                                    fun_V(params, np.linspace(np.min(x_data), np.max(x_data))))
            output['F_stat_v'] = F_stat
            output['p_v'] = p
            y_range = np.max(fun_V(params, x_data)) - np.min(fun_V(params, x_data))
            if (y_range < min_y_range):
                output['F_stat_v'] = -np.inf
                output['p_v'] = np.inf           

    # --- step: 3-parameter fit ---
    # parameters are:
    # step location,
    # left side value,
    # right side value
    if ('step' in requested_shapes):
        fun_step = lambda p, x : np.concatenate([p[1] * np.ones(np.sum(x < p[0])), 
                                                 p[2] * np.ones(np.sum(x >= p[0]))])
        x = np.unique(x_data[~np.isnan(y_data) & x_data != 50])
        mps_default = np.vstack((x[:-1], x[1:])).mean(axis=0)[1:-1]
        if (options is None):
            mps = mps_default
            min_y_range = 0
            n_params = 3
        else:
            idx = np.argmax([dict_['shape'] == 'step' for dict_ in options])
            mps = (options[idx]['midpoints'] if ('midpoints' in options[idx]) else mps_default)
            min_y_range = (options[idx]['min_y_range'] if ('min_y_range' in options[idx]) else 0)
            n_params = (3 if (len(mps) > 1) else 2)
        y_data_clean = y_data[~np.isnan(y_data) & x_data != 50]
        if (len(y_data_clean) < (n_params + 1)) or all((y_data_clean - np.mean(y_data_clean)) == 0.0):
            # not enough data points to fit step...
            output['params_step'] = []
            output['fit_step'] = []
            output['fit_step_full'] = []
            output['F_stat_step'] = -np.inf
            output['p_step'] = np.inf
        else:
            x_data_clean = x_data[~np.isnan(y_data) & x_data != 50]
            SSEs = np.full(len(mps), np.nan)
            all_params = np.full((len(mps), 3), np.nan)
            for i_mp, mp in enumerate(mps):
                params = [mp, np.mean(y_data_clean[x_data_clean < mp]), np.mean(y_data_clean[x_data_clean >= mp])]
                if any(np.isnan(params)):
                    # if either side of step has no data points, skip
                    SSEs[i_mp] = np.inf
                else:
                    SSEs[i_mp] = np.sum((y_data_clean - fun_step(params, x_data_clean)) ** 2)
                all_params[i_mp, :] = params
            if np.isinf(np.min(SSEs)):
                output['params_step'] = []
                output['fit_step'] = []
                output['fit_step_full'] = []
                output['F_stat_step'] = -np.inf
                output['p_step'] = np.inf
            else:
                params = all_params[np.argmin(SSEs), :]
                sse2 = np.sum((y_data_clean - fun_step(params, x_data_clean)) ** 2)
                if (sse2 == 0):
                    print('Warning: step fit to ( x =', x_data_clean, 'y =', y_data_clean, ') is perfect, will raise divide-by-0 warning')
                sse1 = sum((np.mean(y_data_clean) - y_data_clean) ** 2)
                df2 = len(y_data_clean) - n_params
                df1 = n_params - 1
                F_stat = df2 / df1 * (sse1 - sse2) / sse2
                p = scipy.stats.f.sf(F_stat, df1, df2)
                output['params_step'] = params
                output['fit_step'] = fun_step(params, x_data)
                output['fit_step_full'] = (np.linspace(np.min(x_data), np.max(x_data)),
                                           fun_step(params, np.linspace(np.min(x_data), np.max(x_data))))
                output['F_stat_step'] = F_stat
                output['p_step'] = p
                y_range = np.max(fun_step(params, x_data)) - np.min(fun_step(params, x_data))
                if (y_range < min_y_range):
                    output['F_stat_step'] = -np.inf
                    output['p_step'] = np.inf
                    
    return output


def classify_fits(input_dict, alpha=0.01, correction=True, forced_assignment=False, requests=None):

    # parse inputs
    if (requests is None):
        use_all_templates = True
    else:
        use_all_templates = False
        requested_templates = []
        for request in requests:
            if request.lower() in ['linear', 'lin']:
                requested_templates.append('linear')
            elif request.lower() in ['v']:
                requested_templates.append('v')
            elif request.lower() in ['step']:
                requested_templates.append('step')
            else:
                raise ValueError('Non-interpretable request for template {}'.format(request))
    available_templates = [key[7:] for key in input_dict if 'F_stat' in key]
    if use_all_templates:
        templates = available_templates
    else:
        unavailable_templates = [template not in available_templates for template in requested_templates]
        if any(unavailable_templates):
            raise ValueError('Requested template {} was not fitted'.format(
                requested_templates[np.argmax(unavailable_templates)]))
        else:
            templates = requested_templates

    p_val = alpha
    if correction: p_val /= len(templates)
             
    # begin classifying
    output_dict = {key: input_dict[key] for key in input_dict}

    # check if input y_data was constant
    y_data_clean = input_dict['y_data'][~np.isnan(input_dict['y_data'])]
    if all((y_data_clean - np.mean(y_data_clean)) == 0.0):
        output_dict['classified_as'] = 'none'
        output_dict['best_fit'] = np.mean(y_data_clean) * np.ones(len(input_dict['x_data']))
        output_dict['best_fit_full'] = (input_dict['x_data'], np.mean(y_data_clean) * np.ones(len(input_dict['x_data'])))
        return output_dict

    F_stats = [input_dict['F_stat_{}'.format(template)] for template in templates]
    label = templates[np.argmax(F_stats)]

    if label == 'linear':
        if forced_assignment or (input_dict['p_linear'] < p_val):
            if input_dict['params_linear'][0] > 0:
                output_dict['classified_as'] = 'linear increasing'
            else:
                output_dict['classified_as'] = 'linear decreasing'
            output_dict['best_fit'] = input_dict['fit_linear']
            output_dict['best_fit_full'] = input_dict['fit_linear_full']
        else:
            output_dict['classified_as'] = 'none'
            output_dict['best_fit'] = np.mean(y_data_clean) * np.ones(len(input_dict['x_data']))
            output_dict['best_fit_full'] = (input_dict['x_data'], np.mean(y_data_clean) * np.ones(len(input_dict['x_data'])))

    elif label == 'v':
        if forced_assignment or (input_dict['p_v'] < p_val):
            if input_dict['params_V'][0] > 0:
                output_dict['classified_as'] = 'v-shape up'
            else:
                output_dict['classified_as'] = 'v-shape down'
            output_dict['best_fit'] = input_dict['fit_v']
            output_dict['best_fit_full'] = input_dict['fit_v_full']
        else:
            output_dict['classified_as'] = 'none'
            output_dict['best_fit'] = np.mean(y_data_clean) * np.ones(len(input_dict['x_data']))
            output_dict['best_fit_full'] = (input_dict['x_data'], np.mean(y_data_clean) * np.ones(len(input_dict['x_data'])))

    elif label == 'step':
        if forced_assignment or (input_dict['p_step'] < p_val):
            if input_dict['params_step'][2] > input_dict['params_step'][1]:
                output_dict['classified_as'] = 'step up at {}'.format(input_dict['params_step'][0])
            else:
                output_dict['classified_as'] = 'step down at {}'.format(input_dict['params_step'][0])
            output_dict['best_fit'] = input_dict['fit_step']
            output_dict['best_fit_full'] = input_dict['fit_step_full']
        else:
            output_dict['classified_as'] = 'none'
            output_dict['best_fit'] = np.mean(y_data_clean) * np.ones(len(input_dict['x_data']))
            output_dict['best_fit_full'] = (input_dict['x_data'], np.mean(y_data_clean) * np.ones(len(input_dict['x_data'])))

    else:
        raise ValueError('Assigned label {} does not match any options'.format(label))
        
    return output_dict


def parse_label(classification, response_profile_correct, response_profile_error):
    label = classification['classified_as']
    if 'linear' in label:
        return 'Linear'
    elif 'step' in label:
        if (classification['params_step'][0] != 50):
            return 'Step (Perception)'
        else:
            if all(np.isnan(response_profile_error[classification['x_data'] >= 50])) or \
                all(np.isnan(response_profile_error[classification['x_data'] < 50])):
                return 'Step (No Error Trials)'
            else:
                delta = np.nanmean(response_profile_correct[classification['x_data'] >= 50]) - \
                        np.nanmean(response_profile_correct[classification['x_data'] < 50])
                delta_error = np.nanmean(response_profile_error[classification['x_data'] >= 50]) - \
                                np.nanmean(response_profile_error[classification['x_data'] < 50])
                if (delta * delta_error < 0):
                    return 'Step (Choice)'
                else:
                    return 'Step (Perception)'
    elif 'v' in label:
        return 'V'
    else:
        return 'Other'
    
    
def fit_psychometric(x, y, 
                     slope0=0.1, midpoint0=50,
                     lower_asymptote0=0, upper_asymptote0=1,
                     slope_bounds=[0, np.inf], midpoint_bounds=[15, 85],
                     lower_asymptote_bounds=[0, 1], upper_asymptote_bounds=[0, 1]):
    '''
    Fit a psychometric function (4-parameter logistic) to x and y data
    '''
    fun_psycho = lambda x, p0, p1, p2, p3: (p0 - p3) / (1 + np.exp(-p1 * (x - p2))) + p3
    p0 = [upper_asymptote0, slope0, midpoint0, lower_asymptote0]
    x_ = np.linspace(x[0], x[-1])
    try:
        p, _ = scipy.optimize.curve_fit(
            fun_psycho, 
            x, 
            y, 
            p0=p0, 
            bounds=([upper_asymptote_bounds[0], slope_bounds[0], midpoint_bounds[0], lower_asymptote_bounds[0]], 
                    [upper_asymptote_bounds[1], slope_bounds[1], midpoint_bounds[1], lower_asymptote_bounds[1]])
        )
        return {
            'x': x_, 
            'y': fun_psycho(x_, *p),
            'y_hat': fun_psycho(x, *p),
            'slope': p[1],
            'midpoint': p[2],
            'lower_asymptote': p[3],
            'upper_asymptote': p[0],
            'success': True
        }
    except:
        print('Psychometric fitting failed...')
        return {
            'x': x_, 
            'y': fun_psycho(x_, *p0),
            'y_hat': fun_psycho(x, *p0),
            'slope': p0[1],
            'midpoint': p0[2],
            'lower_asymptote': p0[3],
            'upper_asymptote': p0[0],
            'success': False
        }

    
def psychometric_comparison_test(x, Y):
    '''
    Extra-sum-of-squares F test to compare two different best-fit psychometric functions
    Each function is a 4-parameter logistic fit to x and y data
    There is one x for both datasets; Y is a list of the two y datasets
    '''
    
    y1, y2 = Y
    
    # one model for all data
    res_all = fit_psychometric(np.array(list(x) + list(x)), np.array(list(y1) + list(y2)))
    if not res_all['success']: return None
    SSE_1 = ((res_all['y_hat'] - np.array(list(y1) + list(y2))) ** 2).sum()
    df_1 = 2 * len(x) - 4
    
    # two separate models
    res_1 = fit_psychometric(x, y1)
    if not res_1['success']: return None
    res_2 = fit_psychometric(x, y2)
    if not res_2['success']: return None
    SSE_2 = ((res_1['y_hat'] - y1) ** 2).sum() + ((res_2['y_hat'] - y2) ** 2).sum()
    df_2 = 2 * len(x) - 8
    
    F = ((SSE_1 - SSE_2) / SSE_2) / ((df_1 - df_2) / df_2)
    p = scipy.stats.f.sf(F, df_1 - df_2, df_2)
    
    return {'F': F, 'p': p}

    
# def psychometric_comparison_test(x, Y):
    
#     fun_psycho = lambda x, p0, p1, p2, p3: (p0 - p3) / (1 + np.exp(-p1 * (x - p2))) + p3
#     y1, y2 = Y
    
#     # one model for all data
#     try:
#         p0 = [1, 0.1, 50, 0]
#         p_all, _ = scipy.optimize.curve_fit(
#             fun_psycho, 
#             np.array(list(x) + list(x)), 
#             np.array(list(y1) + list(y2)), 
#             p0=p0, 
#             bounds=([0, 0, 15, 0], [1, np.inf, 85, 1])
#         )
#     except:
#         print('Psychometric fitting failed...')
#         return None
#     y_hat_all = fun_psycho(x, *p_all)
#     SSE_1 = ((y_hat_all - y1) ** 2).sum() + ((y_hat_all - y2) ** 2).sum()
#     df_1 = 2 * len(x) - 4
    
#     # two separate models
#     try:
#         p0 = [1, 0.1, 50, 0]
#         p_1, _ = scipy.optimize.curve_fit(
#             fun_psycho, 
#             x, 
#             y1, 
#             p0=p0, 
#             bounds=([0, 0, 15, 0], [1, np.inf, 85, 1])
#         )
#     except:
#         print('Psychometric fitting failed...')
#         return None
#     y_hat_1 = fun_psycho(x, *p_1)
#     try:
#         p0 = [1, 0.1, 50, 0]
#         p_2, _ = scipy.optimize.curve_fit(
#             fun_psycho, 
#             x, 
#             y2, 
#             p0=p0, 
#             bounds=([0, 0, 15, 0], [1, np.inf, 85, 1])
#         )
#     except:
#         print('Psychometric fitting failed...')
#         return None
#     y_hat_2 = fun_psycho(x, *p_2)
#     SSE_2 = ((y_hat_1 - y1) ** 2).sum() + ((y_hat_2 - y2) ** 2).sum()
#     df_2 = 2 * len(x) - 8
    
#     F = ((SSE_1 - SSE_2) / SSE_2) / ((df_1 - df_2) / df_2)
#     p = scipy.stats.f.sf(F, df_1 - df_2, df_2)
    
#     return {'F': F, 'p': p}
    
    
def rmAnova2Way(df):
    '''
    Perform 2-way repeated measures (within subjects) ANOVA
    Requires dataframe input with fields:
        id: the subject id
        iv1: independent variable 1 level for data point
        iv2: independent variable 2 level for data point
        dv: dependent variable data point value
        
    Confirmed to match the output of MATLAB's 'fitrm' and 'ranova'
    '''
    grand_mean = df['dv'].mean()
    SST = np.sum((df['dv'] - grand_mean) ** 2)
    data_1 = np.hstack([np.array(df[df['iv1'] == level]['dv']).reshape([-1, 1]) for level in df['iv1'].unique()])
    SSE1 = data_1.shape[0] * np.sum((data_1.mean(axis=0) - grand_mean) ** 2)
    df_1b = len(df['iv1'].unique()) - 1
    data_2 = np.hstack([np.array(df[df['iv2'] == level]['dv']).reshape([-1, 1]) for level in df['iv2'].unique()]) 
    SSE2 = data_2.shape[0] * np.sum((data_2.mean(axis=0) - grand_mean) ** 2) 
    df_2b = len(df['iv2'].unique()) - 1
    mean_over_subjects = np.array([
        df[(df['iv1'] == level1) & (df['iv2'] == level2)]['dv'].mean() 
        for level2 in df['iv2'].unique() for level1 in df['iv1'].unique()
    ])
    SSE12 = len(df['id'].unique()) * np.sum((mean_over_subjects - grand_mean) ** 2) - SSE1 - SSE2 
    df_12b = df_1b * df_2b
    mean_over_vars = np.array([
        df[df['id'] == id_]['dv'].mean() 
        for id_ in df['id'].unique()
    ])
    SSsub = len(df['iv1'].unique()) * len(df['iv2'].unique()) * np.sum((mean_over_vars - grand_mean) ** 2) 
    SSwithin = np.sum([
        (df['dv'].iloc[i] - df[(df['iv1'] == df['iv1'].iloc[i]) & (df['iv2'] == df['iv2'].iloc[i])]['dv'].mean()) ** 2
        for i in range(len(df))
    ])
    data_1w = [df[(df['iv1'] == df['iv1'].iloc[i]) & (df['id'] == df['id'].iloc[i])]['dv'].mean() 
               for i in range(len(df))]
    SSE1w = np.sum((data_1 - data_1.mean(axis=0)) ** 2) - np.sum((df['dv'] - data_1w) ** 2) - SSsub 
    df_1w = (len(df['id'].unique()) - 1) * df_1b
    data_2w = [df[(df['iv2'] == df['iv2'].iloc[i]) & (df['id'] == df['id'].iloc[i])]['dv'].mean() 
               for i in range(len(df))]
    SSE2w = np.sum((data_2 - data_2.mean(axis=0)) ** 2) - np.sum((df['dv'] - data_2w) ** 2) - SSsub  
    df_2w = (len(df['id'].unique()) - 1) * df_2b
    SSE12w = SSwithin - SSE1w - SSE2w - SSsub 
    df_12w = (len(df['id'].unique()) - 1) * df_12b
    F1 = (SSE1 / df_1b) / (SSE1w / df_1w)
    p1 = scipy.stats.f.sf(F1, df_1b, df_1w)
    F2 = (SSE2 / df_2b) / (SSE2w / df_2w)
    p2 = scipy.stats.f.sf(F2, df_2b, df_2w)
    F12 = (SSE12 / df_12b) / (SSE12w / df_12w)
    p12 = scipy.stats.f.sf(F12, df_12b, df_12w)
    print('SSE_1b = {:.4f}, SSE_1w = {:.4f}, F_1 = {:.4f}, p_1 = {:.4f}'.format(SSE1, SSE1w, F1, p1))
    print('SSE_2b = {:.4f}, SSE_2w = {:.4f}, F_2 = {:.4f}, p_2 = {:.4f}'.format(SSE2, SSE2w, F2, p2))
    print('SSE_12b = {:.4f}, SSE_12w = {:.4f}, F_12 = {:.4f}, p_12 = {:.4f}'.format(SSE12, SSE12w, F12, p12))
    return {
        'SSE_1b': SSE1,
        'SSE_1w': SSE1w,
        'F_1': F1,
        'p_1': p1,
        'SSE_2b': SSE2,
        'SSE_2w': SSE2w,
        'F_2': F2,
        'p_2': p2,
        'SSE_12b': SSE12,
        'SSE_12w': SSE12w,
        'F_12': F12,
        'p_12': p12
    }


def binomial_threshold(alpha, N, chance, n_tails=2):
    alt = 'greater' if (n_tails == 1) else 'two-sided'
    k = N + 1
    p = 0
    while p < alpha:
        k -= 1
        p = scipy.stats.binomtest(k, N, p=chance, alternative=alt).pvalue
    thresh = 100 * (k + 1) / N
    return thresh


def dist_euclidean(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def dist_cosine(x, y):
    norm_x = np.sqrt((x * x).sum())
    norm_y = np.sqrt((y * y).sum())
    if 0 in [norm_x, norm_y]:
        return np.nan
    else:
        return 1 - (x * y).sum() / (norm_x * norm_y)
        
        
def dist_correlation(x, y):
    sigma_x = np.sqrt(((x - x.mean()) ** 2).sum())
    sigma_y = np.sqrt(((y - y.mean()) ** 2).sum())
    if 0 in [sigma_x, sigma_y]:
        return np.nan
    else:
        return 1 - ((x - x.mean()) * (y - y.mean())).sum() / (sigma_x * sigma_y)
    
    
def overlap(u, v):
    norm_u = np.sqrt((u * u).sum())
    norm_v = np.sqrt((v * v).sum())
    if 0 in [norm_u, norm_v]: return np.nan
    return np.abs((u * v).sum()) / norm_u / norm_v
