import matplotlib.pyplot as plt
import numpy as np

# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

def plot_histogram(data, number_of_bins=200, title='Histogram', xlabel='Value', ylabel='Frequency'):
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	#bins = 10 ** (np.arange(-2.0, 0.5, step=0.1))
	#n, bins, patches = \
	plt.hist(data, bins=number_of_bins, facecolor='green', alpha=0.75)
	#plt.xscale('log')
	fig = plt.gcf()
	plt.show()
