import numpy as np
from scipy import spatial
from datamanager import DataManager
from copy import deepcopy

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

class Predictor:

	def __init__(self, training_data, test_data, validation_data,
	             training_labels, test_labels, validation_labels, predictor_func, closeness_cutoff=None):
		self.training_data = training_data
		self.test_data = test_data
		self.validation_data = validation_data
		self.training_labels = training_labels
		self.test_labels = test_labels
		self.validation_labels = validation_labels
		self.predictor_func = predictor_func
		self.closeness_cuttoff = closeness_cutoff

	def GRBT(self):
		# fit estimator
		if (self.test_data == None or self.training_data == None):
			return False
		if (len(self.test_data) == 0):
			return False
		if (len(self.training_data) < 2):
			return False
		label = self.test_labels[0]
		approved = False
		for l in self.test_labels:
			if l != label:
				approved = True
		if not approved: # only one distinct label
			return False
		#print('Length of training data:',len(self.training_data))
		#print('Length of training labels:',len(self.training_labels))
		self.est = GradientBoostingClassifier(n_estimators=200, max_depth=3)
		self.est.fit(self.training_data, self.training_labels)

		# predict class labels
		#pred = self.est.predict(self.test_data)

		# score on test data (accuracy)
		acc = self.est.score(self.test_data, self.test_labels)
		print('ACC: %.4f' % acc)

		# predict class probabilities
		print(self.est.predict_proba(self.training_data)[0])
		return True

	def GRBT_predict(self, case):
		return None, self.est.predict(case)[0]

	def find_best_matching_case(self, case, station_status):
		# end_station must be same for all trip data
		if station_status != None:
			stations_that_have_capacity = []
			stations_that_have_capacity_labels = []
			for i, data_point in enumerate(self.training_data):
				station_number = self.training_labels[i]
				if (get_available_slots(station_status, station_number) > 0):
					stations_that_have_capacity.append(data_point)
					stations_that_have_capacity_labels.append(station_number)
				#original_tree = spatial.KDTree(self.training_data)
				#original_choice = original_tree.query(case)
				#tree = spatial.KDTree(stations_that_have_capacity)
				#best_case = tree.query(case)
				#if (best_case != original_choice):
				#	print(best_case, original_choice)
		else:
			print('WARNING! station status is none')
			stations_that_have_capacity = self.training_data
			stations_that_have_capacity_labels = self.training_labels
		if (len(stations_that_have_capacity) < 2):
			return -1, -1
		tree = spatial.KDTree(stations_that_have_capacity)
		best_case = tree.query(case)
		distance = best_case[0]
		#if (get_available_slots(station_status, best_case[1]) == 0):
		#	print(get_available_slots(station_status, best_case[1]), self.st[best_case[1]])
		print(distance)
		if distance < 999999999:# self.closeness_cuttoff:
			return stations_that_have_capacity[best_case[1]],\
			       stations_that_have_capacity_labels[best_case[1]]
		return -1, -1

	def prediction_of_label_by_best_matching_case(self, case, station_status):
		best_case, best_case_label = self.find_best_matching_case(case=case, station_status=station_status)
		return best_case, best_case_label

	def run_all_test_data(self):
		hits_misses = [0, 0]
		for i, case in enumerate(self.test_data):
			_, label_prediction= self.get_prediction(case=case)
			correct_label = self.test_labels[i]
			#print(label_prediction, correct_label)
			if label_prediction == correct_label:
				hits_misses[0] += 1
			else:
				hits_misses[1] += 1
		print("Run all test_data for single user")
		print(hits_misses)

	def get_prediction(self, case, station_status):
		if self.predictor_func == "BestMatchingCase":
			return self.prediction_of_label_by_best_matching_case(
				case=case, station_status=station_status)
		elif self.predictor_func == "GRBT":
			return self.GRBT_predict(case=case)
		else:
			print("Didn't recognize predictor function: "+self.predictor_func)
			return self.prediction_of_label_by_best_matching_case(
				case=case, station_status=station_status)

def get_available_slots(station_status, station_number):
	if (station_number in station_status.keys()):
		return station_status[station_number]
	else: return 0