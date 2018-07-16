import numpy as np
import pandas as pd
import re
import pprint
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split


def data_clean(df):
    df.drop([0,1], inplace=True)
    df.columns = [col.lower().replace('.', '') for col in df.columns]
    df.columns = [col.rstrip().replace(' -', '') for col in df.columns]
    df.columns = [re.sub(r'\s\(\w*\/*\w*\)', '', col) for col in df.columns]
    df.columns = [col.replace('average porosity, ', '') for col in df.columns]
    df.columns = [col.replace(' ', '_') for col in df.columns]
    df.columns = [col.replace('perforations', 'perfs') for col in df.columns]

    return_df = df.loc[:, ['api12', 'sh_latitude', 'sh_longitude',
                           'completion_date', 'producing_formation', 'ip30', 'ip180',
                           'ip365', 'cum_gas_6_months', 'number_of_frac_stages',
                           'total_proppant_pumped',
                           'total_fluid_pumped', 'td', 'lewis_a001_perfs',
                           'lewis_a002_perfs', 'lewis_a003_perfs',
                           'lewis_b_clino_perfs', 'lewis_c_clino_perfs',
                           'lewis_d_clino_perfs', 'lewis_e_clino_perfs',
                           'lewis_f_clino_perfs', 'lewis_g_clino_perfs',
                           'lewis_h_clino_perfs', 'upper_almond_bar_perfs',
                           'upper_almond_marine_perfs', 'middle_almond_non-marine_perfs',
                           'middle_almond_marine_wedge_perfs', 'lower_almond_non-marine_perfs',
                           'lower_almond_fluvial_perfs', 'shallowest_perfs',
                           'deepest_perfs', 'upper_almond_bar_phia',
                           'upper_almond_bar_phih', 'upper_almond_marine_phia',
                           'upper_almond_marine_phih', 'middle_almond_non-marine_phia',
                           'middle_almond_non-marine_phih', 'middle_almond_marine_wedge_phia',
                           'middle_almond_marine_wedge_phih', 'lower_almond_non-marine_phia',
                           'lower_almond_non-marine_phih', 'lower_almond_fluvial_phia',
                           'lower_almond_fluvial_phih']]

    return_df.rename(str, columns={'sh_longitude': 'lon', 'sh_latitude': 'lat'},
                     inplace=True)

    def type_convert(col, d_type):
        return_df.loc[:,col] = return_df.loc[:,col].astype(d_type)

    def perfs_to_int(col):
        return_df.loc[:,col] = return_df.loc[:,col].map(dict(Y=1, N=0))
        return_df.loc[:,col].fillna(0, inplace=True)
        type_convert(col, int)

    for col in return_df.columns:
        if 'perfs' in col:
            perfs_to_int(col)
        elif 'deep' in col or 'shallow' in col or 'ip' in col or 'cum' in col \
            or 'phia' in col or 'phih' in col:
            type_convert(col, float)
        elif col in ['lat', 'lon', 'api12', 'td']:
            type_convert(col, float)
        elif col in ['number_of_frac_stages', 'total_proppant_pumped',
                     'total_fluid_pumped']:
            return_df.loc[:,col] = pd.to_numeric(return_df.loc[:,col],
                                                 errors='coerce')
        elif 'date' in col:
            return_df.loc[:,col] = pd.to_datetime(return_df.loc[:,col])

    return_df = return_df.loc[((return_df['upper_almond_bar_perfs'] != 0) |
                              (return_df['upper_almond_marine_perfs'] != 0) |
                              (return_df['middle_almond_non-marine_perfs'] != 0) |
                              (return_df['middle_almond_marine_wedge_perfs'] != 0) |
                              (return_df['lower_almond_non-marine_perfs'] != 0) |
                              (return_df['lower_almond_fluvial_perfs'] != 0)) &
                              (return_df['completion_date'].notnull()) &
                              ((return_df['producing_formation'] == 'MESAVERDE') |
                              (return_df['producing_formation'] == 'ALMOND')) &
                              (return_df['number_of_frac_stages'] > 0) &
                              (return_df['total_proppant_pumped'] > 0) &
                              (return_df['ip30'] > 0) &
                              (return_df['ip180'] > 0) &
                              (return_df['ip365'] > 0),
                              :]

    for col in return_df.columns:
        if 'lewis' in col:
            return_df = return_df.loc[return_df[col] == 0, :]
            return_df.drop(col, axis=1, inplace=True)

    return_df.fillna(0, inplace=True)

    return return_df

def forest_model(df, pred_label='ip30'):
    model_df = df[['lat', 'lon', 'total_proppant_pumped', 'total_fluid_pumped',
                   'td',  'upper_almond_bar_perfs', 'upper_almond_marine_perfs',
                   'middle_almond_non-marine_perfs', 'middle_almond_marine_wedge_perfs',
                   'lower_almond_non-marine_perfs', 'lower_almond_fluvial_perfs',
                   'shallowest_perfs', 'deepest_perfs', 'upper_almond_bar_phia',
                   'upper_almond_bar_phih', 'upper_almond_marine_phia',
                   'upper_almond_marine_phih', 'middle_almond_non-marine_phia',
                   'middle_almond_non-marine_phih', 'middle_almond_marine_wedge_phia',
                   'middle_almond_marine_wedge_phih', 'lower_almond_non-marine_phia',
                   'lower_almond_non-marine_phih', 'lower_almond_fluvial_phia',
                   'lower_almond_fluvial_phih']]

    y = df[pred_label].values

    X_train, X_test, y_train, y_test = train_test_split(model_df.values, y,
                                                        test_size=0.2,
                                                        random_state=13)
    rf = RandomForestRegressor(random_state=20)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)

    best_label = 'ip365'
    best = 0.163

    print('Best score for random forest regressor on {}: {}'.format(best_label, best))
    print('-------------------------------------------------')
    print('Score for current model predicting for {}: {}'.format(pred_label, rf.score(X_test, y_test)))
    print('\n')

    return pred

def boosted_model(df, pred_label='ip30'):
    model_df = df[['lat', 'lon', 'total_proppant_pumped', 'total_fluid_pumped',
                   'td',  'upper_almond_bar_perfs', 'upper_almond_marine_perfs',
                   'middle_almond_non-marine_perfs', 'middle_almond_marine_wedge_perfs',
                   'lower_almond_non-marine_perfs', 'lower_almond_fluvial_perfs',
                   'shallowest_perfs', 'deepest_perfs', 'upper_almond_bar_phia',
                   'upper_almond_bar_phih', 'upper_almond_marine_phia',
                   'upper_almond_marine_phih', 'middle_almond_non-marine_phia',
                   'middle_almond_non-marine_phih', 'middle_almond_marine_wedge_phia',
                   'middle_almond_marine_wedge_phih', 'lower_almond_non-marine_phia',
                   'lower_almond_non-marine_phih', 'lower_almond_fluvial_phia',
                   'lower_almond_fluvial_phih']]

    y = df[pred_label].values

    X_train, X_test, y_train, y_test = train_test_split(model_df.values, y,
                                                        test_size=0.2,
                                                        random_state=13)
    gbr = GradientBoostingRegressor(loss='huber',
                                    learning_rate=0.1,
                                    n_estimators=100,
                                    max_depth=10,
                                    max_features=15,
                                    warm_start=False,
                                    random_state=20)
    gbr.fit(X_train, y_train)
    pred = gbr.predict(X_test)
    score = gbr.score(X_test, y_test)

    # gbr = GradientBoostingRegressor(loss='huber',
    #                                 learning_rate=0.1,
    #                                 n_estimators=100,
    #                                 max_depth=10,
    #                                 max_features=15,
    #                                 warm_start=False,
    #                                 random_state=20)

    print('\n')
    print('Best score for gradient boosted regressor: {}'.format(max(best_dic.values())))
    print('------------------------------------------------')
    print('Score for current model predicting for {}: {}'.format(pred_label, score))

    return pred, score


if __name__ == '__main__':
    g_df = pd.read_csv('data/wamda.csv', encoding='ISO-8859-1', low_memory=False)

    clean_df = data_clean(g_df)

    best_dic = {'ip30': 0.158, 'ip180': 0.269, 'ip365': 0.309}
    for pred_lab in ['ip30', 'ip180', 'ip365']:
        # pred = forest_model(clean_df, pred_lab)
        pred, score = boosted_model(clean_df, pred_lab)
        if round(score, 3) > best_dic[pred_lab]:
            best_dic[pred_lab] = round(score, 3)

    print('\n')
    pprint.pprint(best_dic, width=1)
