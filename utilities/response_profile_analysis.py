import numpy as np
import scipy


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