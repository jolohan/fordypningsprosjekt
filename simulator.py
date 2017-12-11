from predictor import Predictor
from extract_config import ExtractConfig
import datamanager
from os import listdir
from os.path import isfile, join
import random as rnd
import csv
import datetime
import numpy as np
import utm

class Simulator():

	def __init__(self, training_size_proportion=0.8, test_size_proportion=0.1,
	             fraction_of_users=1.0, predictor_func='best_matching_case',
	             normalize_data=True):
		self.predictors = {}
		self.users = load_all_users()
		self.training_size_proportion = training_size_proportion
		self.test_size_proportion = test_size_proportion
		self.validation_size_proportion = 1.0 - training_size_proportion - test_size_proportion
		self.fraction_of_users = fraction_of_users
		self.predictor_func = predictor_func
		self.normalize_data = normalize_data
		self.station_coordinates = load_all_station_coordinates()

	def connect_all_users_to_predictors(self, users):
		for i, user in enumerate(users):
			if (i%100 == 0):
				print(i, "/", len(users))
			self.connect_single_user_to_predictor(user_ID=user)

	def connect_single_user_to_predictor(self, user_ID):
		data_manager = datamanager.DataManager(user_ID=user_ID,
		                                       training_days=self.training_days,
		                                       test_days=self.test_days,
		                                       validation_days=self.validation_days,
		                                       normalize_data=self.normalize_data,
		                                       station_coordinates=self.station_coordinates)
		data_manager.set_training_test_validation_trips()
		predictor = Predictor(training_data=data_manager.training_trips,
		                      test_data=data_manager.test_trips,
		                      validation_data=data_manager.validation_trips,
		                      training_labels=data_manager.training_labels,
		                      test_labels=data_manager.test_labels,
		                      validation_labels=data_manager.validation_labels,
		                      predictor_func=self.predictor_func)
		self.predictors[(int)(user_ID)] = predictor

	def load_day(self, day):
		path = 'data/trips_by_day/day_'
		day_string = datamanager.convert_date_to_string(day)
		formatted_csv_result = datamanager.format_csv_result(filename=(path+day_string))
		data, labels = datamanager.format_trip_query_to_data(
			query_job_result=formatted_csv_result,
			station_coordinates=self.station_coordinates)
		return data, labels

	def simulate_day(self, day="20.04.2017"):
		hits_misses_nonguess = [0, 0, 0]
		data, labels = self.load_day(day=day)
		print("Loaded day: " + str(day))
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

	def test(self):
		print('Loading test days')
		hits_misses_nonguess = [0, 0, 0]
		for day in self.test_days:
			data, labels = self.load_day(day=day)
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
		total_guesses = 1.0 * (hits_misses_nonguess[0] + hits_misses_nonguess[1])
		if total_guesses != 0:
			print("Hit percentage: " + str((hits_misses_nonguess[0] / (total_guesses))))
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
	def query_all_unique_days_in_trips(self, load_from_local=False, load_from_dir=True):
		unique_days = []
		if load_from_dir:
			mypath = 'data/trips_by_day/'
			onlyfiles = [f.split('_')[1].split('.')[:3] for f in listdir(mypath) if isfile(join(mypath, f))][1:]
			for row in onlyfiles:
				row_split = [(int)(cell) for cell in row]
				date = datetime.date(year=row_split[2], month=row_split[1], day=row_split[0])
				unique_days.append(date)
		elif (load_from_local):
			with open('query_results/all_trip_dates.csv', 'rt') as csvfile:
				spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
				days = [row[0] for row in spamreader][1:]
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
		training_days = []
		test_days = []
		validation_days = []
		counters = [0, 0, 0]
		for day in self.all_days:
			number = rnd.random()
			if (number<self.training_size_proportion):
				training_days.append(day)
				counters[0] += 1
			elif (number<self.training_size_proportion+self.test_size_proportion):
				test_days.append(day)
				counters[1] += 1
			else:
				validation_days.append(day)
				counters[2] += 1
		self.training_days = np.array(training_days)
		self.test_days = np.array(test_days)
		print(self.test_days[0])
		self.validation_days = np.array(validation_days)
		#print("Counters: ", counters)

	def run(self):
		self.split_days_into_training_test_vaildation()
		percentage_of_users = (int)(len(self.users) * self.fraction_of_users)
		self.connect_all_users_to_predictors(users=self.users[:percentage_of_users])
		#test_day = self.test_days[(int)(len(self.test_days)/3)]
		#self.simulate_day(test_day)
		self.test()

def load_all_station_coordinates():
	station_coordinates = {}
	filename = 'data/stations_2017.csv'
	all_coordinates = []
	all_stations = []
	with open(filename, 'r') as file:
		rows = file.readlines()[2:]
		for row in rows:
			row_split = row.split(';')
			station_number = (int)(row_split[0])
			all_stations.append(station_number)
			lat = (float)(row_split[3])
			lon = (float)(row_split[4])
			utm_coordinates = [utm_coordinate for utm_coordinate in utm.from_latlon(lat, lon)[:2]]
			all_coordinates.append(utm_coordinates)
	all_coordinates = np.array(all_coordinates)
	all_coordinates = datamanager.normalize_data(all_coordinates)
	for i, station in enumerate(all_stations):
		utm_coordinates = all_coordinates[i]
		station_coordinates[station] = utm_coordinates
	return station_coordinates

def load_all_users():
	mypath = 'data/trips_by_user/'
	onlyfiles = [f.split('_')[1].split('.')[0] for f in listdir(mypath) if isfile(join(mypath, f))]
	return onlyfiles[1:]