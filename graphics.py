import matplotlib.pyplot as plt
import numpy as np
import datetime
import os

# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

def plot_histogram(data, number_of_bins=1000, title='Histogram', xlabel='Value', ylabel='Frequency', xscale='linear',
                   save=False, show=True, path='plots/'):
	plt.clf()
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	#bins = 10 ** (np.arange(-2.0, 0.5, step=0.1))
	#n, bins, patches = \
	#bins = np.arange(0, 1.5, step=0.1)
	#bins += np.arange(1.5, 5, step=0.5)
	if number_of_bins == 24:
		number_of_bins = np.arange(0, 24, step=1)
	colors = ['green', 'blue', 'red', 'yellow', 'black']
	labels = ['Predicted', 'Actual']
	for i, single_data in enumerate(data):
		plt.hist(single_data, bins=number_of_bins, facecolor=colors[i], alpha=0.5, label=labels[i])
	plt.legend(loc='upper right')
	plt.xscale(xscale)
	#plt.xticks(10**(np.arange(-2.0, 1, step=0.2)))
	date_string = '_'.join(str(datetime.datetime.now()).split(' ')).split('.')[0]
	#print(date_string)
	if save:
		directory = path + date_string.split(':')[0]
		if not os.path.exists(directory):
			os.makedirs(directory)
		directory += '/'
		filename = directory+title
		#print(filename)
		plt.savefig(filename, bbox_inches='tight')
	#plt.savefig(title, bbox_inches='tight')
	if show:
		plt.show()
