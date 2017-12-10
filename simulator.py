from trainer import Predictor
import datamanager
from os import listdir
from os.path import isfile, join
import random as rnd
import csv
import datetime
import numpy as np

class Simulator():



	def __init__(self, training_size_proportion=0.8, test_size_proportion=0.1):
		self.predictors = {}
		self.users = get_all_users()
		self.training_size_proportion = training_size_proportion
		self.test_size_proportion = test_size_proportion
		self.validation_size_proportion = 1.0 - training_size_proportion - test_size_proportion

	def connect_all_users_to_predictors(self, users):
		for i, user in enumerate(users):
			if (i%100 == 0):
				print(i, "/", len(users))
			self.connect_single_user_to_predictor(user_ID=user)

	def connect_single_user_to_predictor(self, user_ID):
		data_manager = datamanager.DataManager(user_ID,
		                                       self.training_days,
		                                       self.test_days,
		                                       self.validation_days)
		data_manager.set_normalized_trips_for_user()
		data_manager.set_training_test_validation_trips()
		predictor = Predictor(training_data=data_manager.training_trips,
		                      test_data=data_manager.test_trips,
		                      validation_data=data_manager.validation_trips,
		                      training_labels=data_manager.training_labels,
		                      test_labels=data_manager.test_labels,
		                      validation_labels=data_manager.validation_labels)
		self.predictors[(int)(user_ID)] = predictor

	def load_day(self, day):
		path = 'data/trips_by_day/day_'
		print("Loaded day: "+str(day))
		day_string = datamanager.convert_date_to_string(day)
		formatted_csv_result = datamanager.format_csv_result(filename=(path+day_string))
		data, labels = datamanager.format_trip_query_to_data(formatted_csv_result)
		return data, labels

	def simulate_day(self, day="20.04.2017"):
		data, labels = self.load_day(day=day)
		hits_misses_nonguess = [0, 0, 0]
		for i, data_point in enumerate(data):
			correct_label = labels[i]
			pred_label = self.get_prediction(data_point=data_point)
			if pred_label == -1:
				hits_misses_nonguess[2] += 1
			elif (correct_label == pred_label):
				hits_misses_nonguess[0] += 1
			else:
				hits_misses_nonguess[1] += 1
		print("Hits vs misses:", hits_misses_nonguess)
		total_guesses = 1.0*(hits_misses_nonguess[0]+hits_misses_nonguess[1])
		if total_guesses != 0:
			print("Hit percentage: " + str((hits_misses_nonguess[0]/(total_guesses))))
		else:
			print("Didn't guess any")

	def get_prediction(self, data_point):
		# if we can get user prediction:
		user_ID = data_point[-1]
		#print(user_ID, self.predictors.keys())
		if user_ID in self.predictors.keys():
			_, pred_label = self.predictors[user_ID].get_prediction(case=data_point)
			return pred_label
		# else: say we dont want to guess
		return -1

	# Find all unique days in trips so they can be split into training/test
	def query_all_unique_days_in_trips(self, load_from_local=True):
		unique_days = None
		if (load_from_local):
			with open('query_results/all_trip_dates.csv', 'rt') as csvfile:
				spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
				days = [row[0] for row in spamreader][1:]
				unique_days = []
				for row in days:
					row_split = row.split('-')
					row_split = [(int)(cell) for cell in row_split]
					row = datetime.date(row_split[0], row_split[1], row_split[2])
					unique_days.append(row)
		else:
			query_text = 'SELECT CAST(started_at AS DATE) ' \
						 'FROM `uip-students.oslo_bysykkel_legacy.trip` T ' \
						 'GROUP BY CAST(started_at AS DATE) ' \
						 'LIMIT 10000'
			print("collecting query result...")
			query_result = self.query(query_text)
			print("got query result")
			#days_matrix = [self.split_date_time_object(date[0], remove_time=True) for date in query_result]
			unique_days = datamanager.return_first_column_of_query_result(query_result)
		unique_days.sort()
		return unique_days

	def split_days_into_training_test_vaildation(self):
		self.all_days = self.query_all_unique_days_in_trips()
		self.training_days = []
		self.test_days = []
		self.validation_days = []
		counters = [0, 0, 0]
		for day in self.all_days:
			number = rnd.random()
			if (number<self.training_size_proportion):
				self.training_days.append(day)
				counters[0] += 1
			elif (number<self.training_size_proportion+self.test_size_proportion):
				self.test_days.append(day)
				counters[1] += 1
			else:
				self.validation_days.append(day)
				counters[2] += 1
		self.training_days = np.array(self.training_days)
		self.test_days = np.array(self.test_days)
		self.validation_days = np.array(self.validation_days)
		#print("Counters: ", counters)

	def run(self):
		self.split_days_into_training_test_vaildation()
		percentage_of_users = (int)(len(self.users) * 0.001)
		self.connect_all_users_to_predictors(users=self.users[:percentage_of_users])
		test_day = self.test_days[(int)(len(self.test_days)/3)]
		self.simulate_day(test_day)

def get_all_users():
	mypath = 'data/trips_by_user/'
	onlyfiles = [f.split('_')[1].split('.')[0] for f in listdir(mypath) if isfile(join(mypath, f))]
	return onlyfiles[1:]

if __name__ == '__main__':
	simulator = Simulator()
	simulator.run()