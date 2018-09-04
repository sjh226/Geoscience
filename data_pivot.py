import pandas as pd
import numpy as np


def pivot(file_path):
    with open(file_path) as file:
        sect = 0
        df = pd.DataFrame()
        i=0
        for idx, line in enumerate(file):
            row = line.split(',')
            row = [entry.rstrip('\n').strip('"') for entry in row]
            if row[0] == 'UWI':
                sect = idx
                print(len(row))
                if sect != 0:
                    new_cols = {col: col + '_' + df_sect['ZONE NAME'].unique()[0]
                                for col in df_sect.columns if col != 'UWI'}
                    df_sect.rename(index=str, columns=new_cols, inplace=True)
                    if df.empty:
                        df = df.append(df_sect)
                    else:
                        df = df.merge(df_sect)
                df_sect = pd.DataFrame(columns=row)
                cols = row
            if sect != idx:
                # try:
                df_sect = df_sect.append(pd.DataFrame(np.array(row[:len(cols)]).reshape(1, -1), columns=cols))
                # except ValueError:
                #     print(row)
                #     i += 1
                #     if i > 20:
                #         break
        return df


if __name__ == '__main__':
    df = pivot('data/alldata.txt')

    df.to_csv('data/all_pivot.csv')
