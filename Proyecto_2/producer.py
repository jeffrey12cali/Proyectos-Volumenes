import pandas as pd
import time
import shutil

def produce():
	url = "dataset.csv"
	data = pd.read_csv(url,sep = ";",header=0, na_values="?")
	meses = list()
	for i in range(4,9):
		name = r'data/dataframe {}.csv'.format(i)
		print(name)
		data[data.month == i].to_csv(name, index = False, header=True)
		shutil.copy(name,"tmp/")
		time.sleep(50)

produce()
