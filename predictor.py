import numpy as np
from scipy import spatial
from sklearn import ensemble, linear_model
from sklearn.ensemble.tests.test_gradient_boosting import boston

from datamanager import DataManager
from copy import deepcopy

import matplotlib.pyplot as plt
from itertools import islice
from sklearn.metrics import mean_squared_error

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

	def train(self):
		if self.predictor_func == "BestMatchingCase":
			pass
		elif self.predictor_func[:len('Regression')] == "Regression":
			return self.regressor()
		elif self.predictor_func == 'GRBT':
			return self.GRBT_classifier()
		else:
			print("Didn't recognize predictor function: "+self.predictor_func)
			print('Use regresion')
			return self.regressor()
		return True

	def regressor(self):
		if not self.has_data():
			return False
		#if not (check_if_distinct_values(self.training_labels)):
		#	return False
		#if not (check_if_distinct_values(self.test_labels)):
		#	return False

		# multivariate input
		X = self.training_data
		# multivariate output
		Y = self.training_labels
		which_regressor = self.predictor_func[len('Regression'):]
		if (which_regressor == 'Lasso'):
			# Lasso
			self.clf = linear_model.Lasso()
		elif (which_regressor == 'ElasticNet'):
			# ElasticNet
			self.clf = linear_model.ElasticNet()
		elif (which_regressor == 'Linear'):
			self.clf = linear_model.LinearRegression()
		else:
			print("Didnt recognize regressorfunction: "+which_regressor+". Run: Lasso")
			self.clf = linear_model.Lasso()
		self.clf.fit(X, Y)
		self.clf.predict(self.test_data)


		#print(self.clf.coef_)
		#print(self.clf.score(self.test_data, self.test_labels))
		return True

	def regression_predict(self, case):
		#print("getting prediction")
		return self.clf.predict(case.reshape(1, -1))[0]

	def GRBT_predict(self, case):
		return self.est.predict(case.reshape(1, -1))

	def has_data(self):
		if (type(self.test_data) == None or type(self.training_data) == None):
			return False
		if (len(self.test_data) == 0):
			return False
		if (len(self.training_data) < 2):
			return False
		return True

	def GRBT_classifier(self):
		if self.has_data():
			pass
		else:
			return False
		if check_if_distinct_values(self.training_labels):
			pass
		else:
			return False
		# fit estimator
		#print('Length of training data:',len(self.training_data))
		#print('Length of training labels:',len(self.training_labels))
		self.est = GradientBoostingClassifier(n_estimators=200, max_depth=3) # depth 4-6
		self.est.fit(self.training_data, self.training_labels)

		# predict class labels
		#pred = self.est.predict(self.test_data)

		# score on test data (accuracy)
		acc = self.est.score(self.test_data, self.test_labels)
		print('ACC: %.4f' % acc)

		# predict class probabilities
		print(self.est.predict_proba(self.training_data)[0])
		return True

	def find_best_matching_case(self, case, station_status):
		# end_station must be same for all trip data
		"""if station_status != None:
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
			return -1, -1"""
		stations_that_have_capacity = self.training_data
		stations_that_have_capacity_labels = self.training_labels

		tree = spatial.KDTree(stations_that_have_capacity)
		best_case = tree.query(case)
		distance = best_case[0]
		#if (get_available_slots(station_status, best_case[1]) == 0):
		#	print(get_available_slots(station_status, best_case[1]), self.st[best_case[1]])
		if distance < self.closeness_cuttoff:# self.closeness_cuttoff:
			return stations_that_have_capacity_labels[best_case[1]]
		return [-1, -1]

	def prediction_of_label_by_best_matching_case(self, case, station_status):
		best_case_label = self.find_best_matching_case(case=case, station_status=station_status)
		return best_case_label

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
		elif self.predictor_func[:len('Regression')] == "Regression":
			return self.regression_predict(case=case)
		elif self.predictor_func == 'GRBT':
			return self.GRBT_predict(case=case)
		else:
			print("Didn't recognize predictor function: "+self.predictor_func)
			print("Using regression_prediction")
			return self.regression_predict(case=case)

	def plot_data(self, x_plot, ground_truth_x_plot, figsize=(8, 5)):
		fig = plt.figure(figsize=figsize)
		#gt = plt.plot(x_plot, ground_truth(x_plot), alpha=0.4, label='ground truth')

		# plot training and testing data
		plt.scatter(self.training_data, self.training_labels, s=10, alpha=0.4)
		plt.scatter(self.test_data, self.test_labels, s=10, alpha=0.4, color='red')
		plt.xlim((0, 10))
		plt.ylabel('y')
		plt.xlabel('x')

def get_available_slots(station_status, station_number):
	if (station_number in station_status.keys()):
		return station_status[station_number]
	else: return 0


def check_if_distinct_values(data):
	data_0 = data[0]
	for d in data:
		if d != data_0:
			return True
	return False

def ground_truth(data, labels):
	return labels