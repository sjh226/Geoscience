import pandas as pd
import numpy as np
import re
import os
import pyodbc
import sys


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
					df_sect.to_csv('data/all/data_{}.csv'.format(sect))
					# if df.empty:
					#     df = df.append(df_sect)
					# else:
					#     df = df.merge(df_sect)
				df_sect = pd.DataFrame(columns=row)
				cols = row
			if sect != idx:
				df_sect = df_sect.append(pd.DataFrame(np.array(row).reshape(1, -1),
										 columns=cols))

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

def sql_push(df, sect):
	connection = pyodbc.connect(r'DRIVER={SQL Server Native Client 11.0};'
								r'SERVER=SQLDW-L48.BP.COM;'
								r'DATABASE=TeamOperationsAnalytics;'
								r'trusted_connection=yes;'
								r'APP=Petra Write'
								)

	df.columns = [col.replace(' ', '_') for col in df.columns]
	col_type = ['varchar(255)'] * (len(df.columns) - 1)

	cols = zip(df.columns[1:], col_type)

	form = ''
	for col, typ in list(cols):
		form += col + ' ' + typ + ', '
	form = form.rstrip(', ').replace(']_', '').replace('[', '').replace('-_', '')

	cursor = connection.cursor()

	SQLCommand = ("""
		DROP TABLE IF EXISTS TeamOperationsAnalytics.dbo.Petra_{0}
	""".format(sect))

	cursor.execute(SQLCommand)
	connection.commit()


	SQLCommand = ("""
		CREATE TABLE TeamOperationsAnalytics.dbo.Petra_{0} (
			{1}
		);
	""".format(sect, form))
	print(form)
	cursor.execute(SQLCommand)
	connection.commit()

	df.fillna('NULL', inplace=True)

	for idx, row in df.iterrows():
		vals = [row[col] for col in df.columns[1:]]
		string_cols = '{}'.format(tuple(df.columns[1:].values)).replace("'", "").replace(']', '').replace('[', '').replace('-_', '')

		SQLCommand = ("""
			INSERT INTO TeamOperationsAnalytics.dbo.Petra_{0} {1}
			VALUES {2}
		""".format(sect, string_cols, tuple(vals)))

		cursor.execute(SQLCommand)
		connection.commit()

	# Close SQL connection once all rows are written
	connection.close()


if __name__ == '__main__':
	df = pivot('data/alldata.txt')

	# df = dataframe_merge('data/all')

	folder = 'data/all'
	for filename in os.listdir(folder):
		if filename.endswith('.csv'):
			df = pd.read_csv(folder + '/' + filename)
			sect = filename.lstrip('data_').rstrip('.csv')
			if sect not in ['106139', '115788', '125437', '135086']:
				sql_push(df, sect)
