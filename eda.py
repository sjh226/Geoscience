import numpy as np
import pandas as pd
import re


def clean_column(col):
    pass


def data_clean(df):
    df.drop([0,1], inplace=True)
    df.columns = [col.lower().replace('.', '') for col in df.columns]
    df.columns = [col.rstrip().replace(' -', '') for col in df.columns]
    df.columns = [re.sub(r'\s\(\w*\/*\w*\)', '', col) for col in df.columns]
    df.columns = [col.replace(' ', '_') for col in df.columns]
    df.columns = [col.replace('perforations', 'perfs') for col in df.columns]

    return_df = df.loc[:, ['api12', 'sh_latitude', 'sh_longitude', 'spud_date',
                           'completion_date', 'producing_formation', 'ip30', 'ip180',
                           'ip365', 'cum_gas_1st_year', 'total_proppant_pumped',
                           'total_fluid_pumped', 'td', 'lewis_a001_perfs',
                           'lewis_a002_perfs', 'lewis_a003_perfs',
                           'lewis_b_clino_perfs', 'lewis_c_clino_perfs',
                           'lewis_d_clino_perfs', 'lewis_e_clino_perfs',
                           'lewis_f_clino_perfs', 'lewis_g_clino_perfs',
                           'lewis_h_clino_perfs', 'upper_almond_bar_perfs',
                           'upper_almond_marine_perfs', 'middle_almond_non-marine_perfs',
                           'middle_almond_marine_wedge_perfs', 'lower_almond_non-marine_perfs',
                           'lower_almond_fluvial_perfs', 'top_of_lewis', 'top_of_fox_hills',
                           'top_of_ericson', 'top_of_upper_almond_bar',
                           'bottom_of_upper_almond_bar', 'top_of_upper_almond_marine',
                           'bottom_of_upper_almond_marine', 'top_of_middle_almond_non-marine',
                           'bottom_of_middle_almond_non-marine', 'top_of_marine_wedge',
                           'bottom_of_marine_wedge', 'top_of_lower_almond_non-marine',
                           'bottom_of_lower_almond_non-marine', 'top_of_lower_almond_fluvial',
                           'bottom_of_lower_almond_fluvial']]

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
        elif 'top' in col or 'bottom' in col:
            type_convert(col, float)
        elif col in ['lat', 'lon', 'api12']:
            type_convert(col, float)
        elif 'date' in col:
            return_df.loc[:,col] = pd.to_datetime(return_df.loc[:,col])

    return return_df


if __name__ == '__main__':
    g_df = pd.read_csv('data/wamda.csv', encoding='ISO-8859-1', low_memory=False)

    clean_df = data_clean(g_df)
