import pandas as pd
import numpy as np


def pivot(file_path):
    with open(file_path) as file:
        sect = 0
        df = pd.DataFrame()
        for idx, line in enumerate(file):
            row = line.split(',')
            row = [entry.rstrip('\n').strip('"') for entry in row]
            if row[0] == 'UWI':
                sect = idx
                if sect != 0:
                    if df.empty:
                        df = df.append(df_sect)
                    else:
                        join = [col for col in row if col in df.columns]
                        df = df.join(df_sect, on=join)
                df_sect = pd.DataFrame(columns=row)
                cols = row
            if sect != idx:
                df_sect = df_sect.append(pd.DataFrame(np.array(row).reshape(1, -1), columns=cols))
        return df


if __name__ == '__main__':
    df = pivot('data/Selecteddata.txt')
