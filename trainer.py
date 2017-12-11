
import numpy as np
from scipy import spatial
from datamanager import DataManager
from copy import deepcopy


class Predictor:

	def __init__(self, training_data, test_data, validation_data,
	             training_labels, test_labels, validation_labels, predictor_func):
		self.training_data = training_data
		self.test_data = test_data
		self.validation_data = validation_data
		self.training_labels = training_labels
		self.test_labels = test_labels
		self.validation_labels = validation_labels
		self.predictor_func = predictor_func

	def find_best_matching_case(self, case):
		# end_station must be same for all trip data
		if (len(self.training_data) > 1):
			tree = spatial.KDTree(self.training_data)
			best_case = tree.query(case)
			distance = best_case[0]
			best_case = best_case[1]
			return best_case
		elif (len(self.training_data) == 1):
			return 0
		else:
			return -1

	def prediction_of_label_by_best_matching_case(self, case):
		best_case = self.find_best_matching_case(case=case)
		if best_case == -1:
			return -1, -1
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
		print("Run all test_data for single user")
		print(hits_misses)

	def get_prediction(self, case):
		if self.predictor_func == "BestMatchingCase":
			return self.prediction_of_label_by_best_matching_case(case=case)
		else:
			print("Didn't recognize predictor function: "+self.predictor_func)
			return self.prediction_of_label_by_best_matching_case(case=case)
