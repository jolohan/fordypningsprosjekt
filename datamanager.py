# Imports the Google Cloud client library
from google.cloud import bigquery
import datetime
import numpy as np
from sklearn import preprocessing
import random as rnd
import csv
import hdbscan

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

	def __init__(self, user_ID, training_days, test_days, validation_days, normalize_data,
	             station_coordinates, cut_off_point_data_amount=0, min_cluster_size=0):
		self.user_ID = user_ID
		# Instantiates a client
		self.bigquery_client = bigquery.Client()
		self.training_days = [convert_to_year_week_weekday(day) for day in training_days]
		self.test_days = [convert_to_year_week_weekday(day) for day in test_days]
		self.validation_days = [convert_to_year_week_weekday(day) for day in validation_days]
		self.cut_off_point_data_amount = cut_off_point_data_amount
		self.normalize_data = normalize_data
		self.station_coordinates = station_coordinates
		self.all_trips = None
		self.all_labels = None
		self.min_cluster_size = min_cluster_size
		#print("all days are to be collected")

	def query(self, query):
		client = bigquery.Client()
		query_job = client.query(query)
		return query_job.result()

	def query_all_trips_by_user(self, userID=5701):
		query_text = 'SELECT * FROM `uip-students.oslo_bysykkel_legacy.trip`' \
				'WHERE member_id = ' + str(userID)
		return self.query(query_text)

	def set_training_test_validation_trips(self):
		self.set_formatted_trips_by_user()
		if (len(self.all_trips) < self.cut_off_point_data_amount):
			return False
		# need central data storage if i am to normalize
		training_trips_index = []
		test_trips_index = []
		validation_trips_index = []
		training_trips = []
		test_trips = []
		validation_trips = []
		training_labels = []
		test_labels = []
		validation_labels = []
		counters = [0, 0, 0]
		for i, trip in enumerate(self.all_trips):
			label = self.all_labels[i]
			year_week_weekday = get_year_week_week_day(trip, min_cluster_size=self.min_cluster_size)
			flag = False
			for test_day in self.test_days:
				if (compare_numpy_dates_arrays(year_week_weekday, test_day)):
					test_trips_index.append(i)
					counters[1] += 1
					flag = True
					break
			if (not flag):
				for validation_day in self.validation_days:
					if (compare_numpy_dates_arrays(year_week_weekday, validation_day)):
						validation_trips_index.append(i)
						counters[2] += 1
						flag = True
						break
				if (not flag):
					training_trips_index.append(i)
					counters[0] += 1
		if self.normalize_data:
			#print("Cant normalize data")
			self.all_trips = normalize_data(self.all_trips)
			#self.all_labels = normalize_data(self.all_labels)
		if (len(training_trips_index) < 1 or
					    len(test_trips_index) + len(validation_trips_index) < 1):
			return False
		self.training_trips = np.array([self.all_trips[index] for index in training_trips_index])
		self.training_labels = np.array([self.all_labels[index] for index in training_trips_index])
		self.test_trips = np.array([self.all_trips[index] for index in test_trips_index])
		self.test_labels = np.array([self.all_labels[index] for index in test_trips_index])
		self.validation_trips = np.array([self.all_trips[index] for index in validation_trips_index])
		self.validation_labels = np.array([self.all_labels[index] for index in validation_trips_index])
		return True

	def set_formatted_trips_by_user(self, user_ID=None, load_from_file=True):
		if user_ID == None:
			user_ID = self.user_ID
		if load_from_file:
			result_from_query_or_csv = format_csv_result('data/trips_by_user/user_' + str(user_ID))
		else:
			result_from_query_or_csv = self.query_all_trips_by_user(userID=user_ID)
		self.all_trips, self.all_labels = format_trip_query_to_data(query_job_result=result_from_query_or_csv,
		                                                            station_coordinates_or_clusters=self.station_coordinates)



# =====================
# == Diverse methods ==
# =====================

def hdb_cluster(data, min_cluster_size=2):
	clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
	#cluster_labels = clusterer.fit(data)
	return clusterer

def normalize_data(data):
	if (len(data) > 0): return preprocessing.scale(data)
	else: return None

def format_csv_result(filename):
	result_from_csv \
		= open(filename + '.csv', 'r').readlines()
	result_from_csv = [row.split(',') for row in result_from_csv]
	result_from_query_or_csv = []
	for row in result_from_csv:
		formatted_row = [cell[1:-1] for cell in row]
		formatted_row[3] = convert_string_to_date(row[3][1:-1])
		formatted_row[4] = convert_string_to_date(row[4][1:-1])
		result_from_query_or_csv.append(formatted_row)
	return result_from_query_or_csv

def format_trip_query_to_data(query_job_result, station_coordinates_or_clusters):
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

		if (start_station == None or (not (start_station in station_coordinates_or_clusters.keys()))):
			pass
			#print(end_station)
			#print(start_station)
		elif (end_station == None or (not (end_station in station_coordinates_or_clusters.keys()))):
			pass
			#print(end_station)
			#print(start_station)
		elif (end_time - start_time) > (datetime.timedelta(minutes=60)):
			#print(end_time, start_time, end_time-start_time)
			pass
		else:
			start_time = split_date_time_object(start_time)
			end_time = split_date_time_object(end_time)
			start_station = station_coordinates_or_clusters[start_station] + [0]
			#print('startstation:' + str(start_station))
			formatted_trip = [coordinate_or_cluster for coordinate_or_cluster in start_station]
			formatted_trip += start_time
			#formatted_trip.append(end_time[-1])
			formatted_trip.append((int)(member_ID))
			formatted_trips.append(formatted_trip)

			end_station = station_coordinates_or_clusters[end_station]
			labels.append(end_station)
	formatted_trips = np.array(formatted_trips)
	labels = np.array(labels)
	return formatted_trips, labels

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

def convert_string_to_date(date_string, delimiter='.', year_index=2, month_index=1, day_index=0):
	date_time_numbers = []
	date_time_split = date_string.split(" ")
	date_split = date_time_split[0].split(delimiter)
	for number in date_split:
		date_time_numbers.append((int)(number))
	time_split = date_time_split[1].split(":")
	for number in time_split:
		date_time_numbers.append((int)(number))
	return datetime.datetime(date_time_numbers[year_index], date_time_numbers[month_index],
	                         date_time_numbers[day_index],
							 date_time_numbers[3], date_time_numbers[4], date_time_numbers[5])

def convert_date_to_string(date):
	day = date.day
	month = date.month
	year = date.year
	s = ""
	date_list = [day, month, year]
	for number in date_list:
		if number < 10:
			s += "0"
		s += str(number)
		s += "."
	s = s[:-1]
	return s

def get_year_week_week_day(trip, min_cluster_size):
	if min_cluster_size < 2:
		return trip[2:5]
	else:
		return trip[1:4]

def convert_date_to_string_year_month_day(date):
	s = convert_date_to_string(date)
	new_s = s[-4:] + '.' + s[-7:-5] + '.' + s[:2]
	return new_s

def compare_numpy_dates_arrays(date1, date2):
	if (date1[0] == date2[0]):
		if (date1[1] == date2[1]):
			if (date1[2] == date2[2]):
				return True
	return False