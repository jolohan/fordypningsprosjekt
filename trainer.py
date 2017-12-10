
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

	def find_best_matching_case(self, case):
		# end_station must be same for all trip data
		tree = spatial.KDTree(self.training_data)
		best_case = tree.query(case)
		distance = best_case[0]
		best_case = best_case[1]
		return best_case

	def prediction_of_label_by_best_matching_case(self, case):
		best_case = self.find_best_matching_case(case=case)
		best_case_label = self.training_labels[best_case]
		return self.training_data[best_case], best_case_label

	def run_all_test_data(self):
		hits_misses = [0, 0]
		for i, case in enumerate(self.test_data):
			_, label_prediction= self.get_prediction(case=case)
			correct_label = self.test_labels[i]
			print(label_prediction, correct_label)
			if label_prediction == correct_label:
				hits_misses[0] += 1
			else:
				hits_misses[1] += 1
		print(hits_misses)

	def get_prediction(self, case):
		return self.prediction_of_label_by_best_matching_case(case=case)
