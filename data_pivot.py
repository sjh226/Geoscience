import pandas as pd
import numpy as np
import re


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
                        df = df.append(df_sect)
                    else:
                        df = df.merge(df_sect)
                df_sect = pd.DataFrame(columns=row)
                cols = row
            if sect != idx:
                df_sect = df_sect.append(pd.DataFrame(np.array(row).reshape(1, -1),
                                         columns=cols))

    # df.dropna(subset=[col for col in df.columns
    #                   if col not in ['UWI', r'ZONE NAME[\w\s]+']],
    #           inplace=True)
    return df


if __name__ == '__main__':
    df = pivot('data/Selecteddata.txt')

    df.to_csv('data/select_pivot.csv')
