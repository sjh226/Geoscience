import pandas as pd
import numpy as np
import re
import os


def pivot(file_path):
    with open(file_path) as file:
        sect = 0
        df = pd.DataFrame()
        for idx, line in enumerate(file):
            row = line.split('","')
            row = [entry.rstrip('\n').strip('"') for entry in row]
            if row[0] == 'UWI':
                sect = idx
                if sect != 0:
                    new_cols = {col: col + '_' + df_sect['ZONE NAME'].unique()[0]
                                for col in df_sect.columns if col != 'UWI'}
                    df_sect.rename(index=str, columns=new_cols, inplace=True)
                    df_sect.replace('', np.nan, inplace=True)
                    df_sect = df_sect.loc[:, df_sect.notnull().sum().gt(df_sect.shape[0] * 0.05)]
                    if df.empty:
                        (df.append(df_sect)).to_csv('data/all/data_{}.csv'.format(sect))
                    else:
                        df = df.merge(df_sect)
                df_sect = pd.DataFrame(columns=row)
                cols = row
            if sect != idx:
                df_sect = df_sect.append(pd.DataFrame(np.array(row).reshape(1, -1),
                                         columns=cols))

    return df

def dataframe_merge(folder):
    full_df = pd.DataFrame()

    for filename in os.listdir(folder):
        if filename.endswith('.csv'):
            df = pd.read_csv(folder + '/' + filename)

            if full_df.empty:
                full_df = full_df.append(df)
            else:
                full_df = full_df.merge(df)

    full_df.to_csv(folder + '_data.csv')

    return full_df


if __name__ == '__main__':
    # df = pivot('data/alldata.txt')
    #
    # df.to_csv('data/all_pivot.csv')

    df = dataframe_merge('data/all')
