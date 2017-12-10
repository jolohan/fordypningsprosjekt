
import numpy as np
from scipy import spatial
from datamanager import DataManager
from copy import deepcopy


class Predictors:

	def __init__(self, training_data, test_data, validation_data,
	             training_labels, test_labels, validation_labels):
		self.training_data = training_data
		self.test_data = test_data
		self.validation_data = validation_data
		self.training_labels = training_labels
		self.test_labels = test_labels
		self.validation_labels = validation_labels

	def find_best_matching_trip(self, trip):
		# end_station must be same for all trip data
		tree = spatial.KDTree(self.training_data)
		best_trip = tree.query(trip)
		distance = best_trip[0]
		best_trip = best_trip[1]
		#print(best_trip, len(self.training_labels))
		return best_trip

	def prediction_of_end_station_by_find_best_matching_trip(self, trip):
		best_trip = self.find_best_matching_trip(trip=trip)
		end_station = self.training_labels[best_trip]
		return self.training_data[best_trip], end_station
