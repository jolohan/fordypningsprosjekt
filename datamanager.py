# Imports the Google Cloud client library
from google.cloud import bigquery
import datetime
import numpy as np
from sklearn import preprocessing
import random as rnd
import csv

#########################################
# export GOOGLE_APPLICATION_CREDENTIALS=/Users/johan/PycharmProjects/Fordypningsprosjekt/'UIP students-8ffa90ab95f7.json'
#########################################

# The name of the datasets


# Queries
query1 = 'SELECT COUNT(DISTINCT (ID)) as trips FROM `uip-students.oslo_bysykkel_legacy.trip` ' \
			'WHERE _PARTITIONTIME >= "2017-11-10 00:00:00" ' \
			'AND _PARTITIONTIME < "2017-11-25 00:00:00" LIMIT 100'


# Prepares a reference to the new dataset
# dataset_ref = bigquery_client.dataset(dataset_id)
# dataset = bigquery.Dataset(dataset_ref)
# print('Dataset {} created.'.format(dataset.dataset_id))



class DataManager():



	def __init__(self, user_ID=5701, training_size_proportion=0.8, test_size_proportion=0.1):
		self.user_ID = user_ID
		self.training_set_size = training_size_proportion
		self.test_set_size = test_size_proportion
		self.validation_set_size = 1.0-training_size_proportion-test_size_proportion
		# Instantiates a client
		self.bigquery_client = bigquery.Client()
		self.all_trips = None
		self.all_labels = None
		self.all_normalized_trips = None
		self.all_normalized_trips_without_end_station = None
		print("all days are to be collected")
		self.all_days = self.query_all_unique_days_in_trips()
		self.split_days_into_training_test_vaildation()

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
			unique_days = return_first_column_of_query_result(query_result)
		unique_days.sort()
		return unique_days

	def split_days_into_training_test_vaildation(self):
		print("splitting training days")
		self.training_days_year_week_weekday = []
		self.test_days_year_week_weekday = []
		self.validation_days_year_week_weekday = []
		#self.training_days = []
		#self.test_days = []
		#self.validation_days = []
		counters = [0, 0, 0]
		for day in self.all_days:
			number = rnd.random()
			if (number<self.training_set_size):
				#self.training_days.append(day)
				self.training_days_year_week_weekday\
					.append(convert_to_year_week_weekday(day))
				counters[0] += 1
			elif (number<self.training_set_size+self.test_set_size):
				#self.test_days.append(day)
				self.test_days_year_week_weekday.append(convert_to_year_week_weekday(day))
				counters[1] += 1
			else:
				#self.validation_days.append(day)
				self.validation_days_year_week_weekday.append(convert_to_year_week_weekday(day))
				counters[2] += 1
		#self.training_days = np.array(self.training_days)
		self.training_days_year_week_weekday = np.array(self.training_days_year_week_weekday)
		self.test_days_year_week_weekday = np.array(self.test_days_year_week_weekday)
		self.validation_days_year_week_weekday = np.array(self.validation_days_year_week_weekday)
		print("Split days into sets")
		print("Counters: ", counters)

	def query(self, query):
		client = bigquery.Client()
		query_job = client.query(query)
		return query_job.result()

	def query_all_trips_by_user(self, userID=5701):
		query_text = 'SELECT * FROM `uip-students.oslo_bysykkel_legacy.trip`' \
				'WHERE member_id = ' + str(userID)
		return self.query(query_text)

	def set_training_test_validation_trips(self):
		self.training_trips = []
		self.test_trips = []
		self.validation_trips = []
		self.training_labels = []
		self.test_labels = []
		self.validation_labels = []
		counters = [0, 0, 0]
		for i, trip in enumerate(self.all_trips):
			label = self.all_labels[i]
			year_week_weekday = trip[1:4]
			flag = False
			for test_trip_day in self.test_days_year_week_weekday:
				if (compare_numpy_dates_arrays(year_week_weekday, test_trip_day)):
					self.test_trips.append(trip)
					self.test_labels.append(label)
					counters[1] += 1
					flag = True
					break
			if (not flag):
				for validation_trip_day in self.validation_days_year_week_weekday:
					if (compare_numpy_dates_arrays(year_week_weekday, validation_trip_day)):
						self.validation_trips.append(trip)
						self.validation_labels.append(label)
						counters[2] += 1
						flag = True
						break
				if (not flag):
					self.training_trips.append(trip)
					self.training_labels.append(label)
					counters[0] += 1


			"""if year_week_weekday in self.training_days_year_week_weekday:
				self.training_trips.append(trip)
				counters[0] += 1
			else:
				counters[1] += 1"""
		self.training_trips = np.array(self.training_trips)
		self.training_labels = np.array(self.training_labels)
		self.test_trips = np.array(self.test_trips)
		self.test_labels = np.array(self.test_labels)
		self.validation_trips = np.array(self.validation_trips)
		self.validation_labels = np.array(self.validation_labels)
		print("Counters: ",counters)

	def set_formatted_trips_by_user(self, user_ID=None, load_from_file=True):
		if user_ID == None:
			user_ID = self.user_ID
		if load_from_file:
			result_from_csv \
				= open('data/trips_by_user/user_' + str(user_ID) + '.csv', 'r').readlines()
			result_from_csv = [row.split(',') for row in result_from_csv]
			result_from_query_or_csv = []
			for row in result_from_csv:
				formatted_row = [cell[1:-1] for cell in row]
				formatted_row[3] = convert_string_to_date(row[3][1:-1])
				formatted_row[4] = convert_string_to_date(row[4][1:-1])
				result_from_query_or_csv.append(formatted_row)
		else:
			result_from_query_or_csv = self.query_all_trips_by_user(userID=user_ID)
		self.all_trips, self.all_labels \
			= self.format_trip_query_to_data(query_job_result=result_from_query_or_csv)


	def format_trip_query_to_data(self, query_job_result):
		trip_data = list(query_job_result)
		formatted_trips = []
		labels = []
		for trip in trip_data:
			member_ID = trip[1]
			if (isinstance(trip[3], int)):
				start_station = trip[2]
				end_station = trip[3]
				start_time = trip[5]
				end_time = trip[6]
			else:
				start_time = trip[3]
				end_time = trip[4]
				start_station = trip[5]
				end_station = trip[6]
			start_station = (int)(start_station)
			end_station = (int)(end_station)

			if end_station == None:
				pass
			else:
				start_time = split_date_time_object(start_time)
				end_time = split_date_time_object(end_time)
				formatted_trip = [start_station]
				formatted_trip += start_time
				formatted_trip.append(end_time[-1])
				formatted_trips.append(formatted_trip)
				labels.append(end_station)
		formatted_trips = np.array(formatted_trips)
		labels = np.array(labels)
		return formatted_trips, labels

	def set_normalized_trips_for_user(self):
		self.set_formatted_trips_by_user()
		self.all_normalized_trips = normalize_data(self.all_trips)
		self.all_normalized_trips_without_end_station = make_copy_deep_at_coulmn_x(self.all_normalized_trips)
		for trip in self.all_normalized_trips_without_end_station:
			trip[1] = 0

def normalize_data(data):
	return preprocessing.scale(data)

def make_copy_deep_at_coulmn_x(twod_numpy_array, deep_copy_index=1):
	new_2d_array = np.copy(twod_numpy_array)
	for i, array in enumerate(twod_numpy_array):
		new_2d_array[i][deep_copy_index] = np.copy(array[deep_copy_index])
	return new_2d_array

def return_first_column_of_query_result(query_result):
	return_array = []
	for row in query_result:
		return_array.append(row[0])
	return return_array

def split_date_time_object(date_time, remove_time=False):
	year = (int)(date_time.year)
	month = (int)(date_time.month)
	day = (int)(date_time.day)
	week = (int)(datetime.date(year, month, day).isocalendar()[1])
	day_of_week = (int)(date_time.weekday())
	if (not remove_time):
		minutes_after_midnight = (int)(date_time.hour*60 + date_time.minute)
		time = [year, week, day_of_week, minutes_after_midnight]
		return time
	#print(year, month, day, week, day_of_week)
	return [year, week, day_of_week]

def convert_to_year_week_weekday(day):
	year = (int)(day.year)
	month = (int)(day.month)
	day_number = (int)(day.day)
	week = (int)(datetime.date(year, month, day_number).isocalendar()[1])
	day_of_week = (int)(day.weekday())
	return [year, week, day_of_week]

def convert_string_to_date(date_string):
	date_time_numbers = []
	date_time_split = date_string.split(" ")
	date_split = date_time_split[0].split(".")
	for number in date_split:
		date_time_numbers.append((int)(number))
	time_split = date_time_split[1].split(":")
	for number in time_split:
		date_time_numbers.append((int)(number))
	return datetime.datetime(date_time_numbers[2], date_time_numbers[1], date_time_numbers[0],
							 date_time_numbers[3], date_time_numbers[4], date_time_numbers[5])

def compare_numpy_dates_arrays(date1, date2):
	if (date1[0] == date2[0]):
		if (date1[1] == date2[1]):
			if (date1[2] == date2[2]):
				return True
	return False


if __name__ == '__main__':
	data_manager = DataManager()
	trips = data_manager.set_normalized_trips_for_user()
