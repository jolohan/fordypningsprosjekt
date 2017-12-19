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
import graphics


class Simulator():
	def __init__(self, training_size_proportion=0.8, test_size_proportion=0.1,
				 fraction_of_users=1.0, predictor_func='best_matching_case',
				 normalize_data=True, cut_off_point_data_amount=10, closeness_cutoff=99999999,
				 error_threshold=0.1, min_cluster_size = 0):
		self.predictors = {}
		self.users = load_all_users()
		rnd.shuffle(self.users)
		self.training_size_proportion = training_size_proportion
		self.test_size_proportion = test_size_proportion
		self.validation_size_proportion = 1.0 - training_size_proportion - test_size_proportion
		self.fraction_of_users = fraction_of_users
		self.predictor_func = predictor_func
		self.normalize_data = normalize_data
		self.cut_off_point_data_amount = cut_off_point_data_amount
		self.closeness_cutoff = closeness_cutoff
		self.station_coordinates_or_clusters = load_all_station_coordinates(min_cluster_size=min_cluster_size)
		if min_cluster_size > 1:
			pass
		else:
			self.inv_station_coordinates = {v[0]: {v[1]: k} for k, v in self.station_coordinates_or_clusters.items()}
		self.error_threshold = error_threshold
		self.min_cluster_size = min_cluster_size

	def connect_all_users_to_predictors(self, users):
		for i, user in enumerate(users):
			if (i % 100 == 0):
				print(i, "/", len(users))
			self.connect_single_user_to_predictor(user_ID=user)

	def connect_single_user_to_predictor(self, user_ID):
		data_manager = datamanager.DataManager(user_ID=user_ID,
		                                       training_days=self.training_days,
		                                       test_days=self.test_days,
		                                       validation_days=self.validation_days,
		                                       normalize_data=self.normalize_data,
		                                       station_coordinates=self.station_coordinates_or_clusters,
		                                       cut_off_point_data_amount=self.cut_off_point_data_amount,
		                                       min_cluster_size=self.min_cluster_size)
		if not data_manager.set_training_test_validation_trips():
			return
		if len(data_manager.all_trips) < self.cut_off_point_data_amount:
			print("Too few trips. Less than: "+str(self.cut_off_point_data_amount))
		predictor = Predictor(training_data=data_manager.training_trips,
		                      test_data=data_manager.test_trips,
		                      validation_data=data_manager.validation_trips,
		                      training_labels=data_manager.training_labels,
		                      test_labels=data_manager.test_labels,
		                      validation_labels=data_manager.validation_labels,
		                      predictor_func=self.predictor_func,
		                      closeness_cutoff=self.closeness_cutoff)
		made_predictor = predictor.train()
		if made_predictor:
			self.predictors[(int)(user_ID)] = predictor

	def load_day(self, day):
		station_status = {}
		for station_number in self.station_coordinates_or_clusters:
			station_status[station_number] = 999999
		path = 'data/trips_by_day/day_'
		day_string = datamanager.convert_date_to_string(day)
		formatted_csv_result = datamanager.format_csv_result(filename=(path + day_string))
		data, labels = datamanager.format_trip_query_to_data(
			query_job_result=formatted_csv_result,
			station_coordinates_or_clusters=self.station_coordinates_or_clusters)
		return data, labels, station_status

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
		total_guesses = 1.0 * (hits_misses_nonguess[0] + hits_misses_nonguess[1])
		if total_guesses != 0:
			print("Hit percentage: " + str((hits_misses_nonguess[0] / (total_guesses))))
		else:
			print("Didn't guess any")

	def test_by_user(self):
		errors = []
		hits_misses_nonguess = [0, 0, 0]
		predicted_vs_correct_arrivals_total = {}
		for user in self.predictors.keys():
			predictor = self.predictors[user]
			data = predictor.test_data
			labels = predictor.test_labels
			for i, data_point in enumerate(data):
				pred_label = predictor.get_prediction(data_point, None)
				if pred_label[0] == -1:
					hits_misses_nonguess[2] += 1
				else:
					correct_label = labels[i]
					day = get_day_number_out_of_trip(trip=data_point, min_cluster_size=self.min_cluster_size)
					minutes = get_minutes_after_midnight_normalized(data_point=data_point)
					if self.min_cluster_size > 1:
						pred_label = pred_label[0]
						closest_vacant_station = pred_label
						#print("closest cluster: "+str(closest_vacant_station))
						correct_station = correct_label
						"""class_prob = predictor.est.predict_proba(data_point.reshape(1, -1))#[0][correct_label]
						if (len(class_prob) > correct_label):
							class_prob += 
						print(class_prob)"""
					else:
						closest_vacant_station = self.find_closest_vacant_station(pred_label)
						correct_station = self.inv_station_coordinates[correct_label[0]][correct_label[1]]
					if not (day in predicted_vs_correct_arrivals_total.keys()):
						predicted_vs_correct_arrivals_total[day] \
							= {station_number: {} for station_number in self.station_coordinates_or_clusters}
					if minutes in predicted_vs_correct_arrivals_total[day][closest_vacant_station].keys():
						predicted_vs_correct_arrivals_total[day][closest_vacant_station][minutes][0] += 1
					else:
						predicted_vs_correct_arrivals_total[day][closest_vacant_station][minutes] = [1, 0]
					if minutes in predicted_vs_correct_arrivals_total[day][correct_station].keys():
						predicted_vs_correct_arrivals_total[day][correct_station][minutes][1] += 1
					else:
						predicted_vs_correct_arrivals_total[day][correct_station][minutes] = [0, 1]
					if self.min_cluster_size < 2:
						error = np.linalg.norm(correct_label - pred_label)
						if error < self.error_threshold:
							hits_misses_nonguess[0] += 1
						else:
							if error > 5:
								print(error, correct_label, pred_label)
							hits_misses_nonguess[1] += 1
						errors.append(error)
					else:
						if (pred_label == correct_label):
							hits_misses_nonguess[0] += 1
						else:
							hits_misses_nonguess[1] += 1

					"""if (self.station_status[correct_label] == 0 or self.station_status[pred_label] == 0):
						print("predicted:", pred_label, '   correct:', correct_label,
							  '\nstatus predicted:', self.station_status[pred_label],
							  'status correct:', self.station_status[correct_label])"""
		print("Hits vs misses:", hits_misses_nonguess)
		total_guesses = 1.0 * (hits_misses_nonguess[0] + hits_misses_nonguess[1])
		total_guesses += 0.001
		hit_percentage = hits_misses_nonguess[0] / (total_guesses)
		if self.min_cluster_size < 2:
			average_error = (np.sum(errors) / total_guesses)
			median_error = np.mean(errors)
		else:
			average_error = 0
			median_error = -1
		if total_guesses != 0:
			print("Hit percentage: %.4f" % hit_percentage)
			if self.min_cluster_size < 2:
				print("Average error: %.3f" % average_error)
		else:
			print("Didn't guess any")
		counter = 0
		for day in predicted_vs_correct_arrivals_total.keys():
			counter += 1
			print("Day:" +str(day) + ". Day # "+str(counter)+ ' out of: '+str(len(predicted_vs_correct_arrivals_total.keys())))
			for station in predicted_vs_correct_arrivals_total[day].keys():
				if len(predicted_vs_correct_arrivals_total[day][station].keys()) != 0:
					if self.min_cluster_size > 1:
						if station < 30:
							plot_day(predicted_vs_correct_arrivals_total, day_key=day, station_key=station,
							         cluster=True)
					else:
						plot_day(predicted_vs_correct_arrivals_total, day_key=day, station_key=station)
		print(hits_misses_nonguess, hit_percentage, average_error, median_error)
		return hits_misses_nonguess, hit_percentage, average_error, errors, median_error

	def find_closest_vacant_station(self, coordinates, station_status=None):
		closest_station = -1
		distance = 99999999
		for station_number in self.station_coordinates_or_clusters.keys():
			station_number_coordinates = self.station_coordinates_or_clusters[station_number]
			new_distance = np.linalg.norm(coordinates-station_number_coordinates)
			if new_distance < distance:
				if station_status == None:
					distance = new_distance
					closest_station = station_number
				else:
					if self.station_status[station_number] != 0:
						distance = new_distance
						closest_station = station_number

		return closest_station

	def test(self):
		errors = []
		print('Loading test days')
		hits_misses_nonguess = [0, 0, 0]
		predicted_vs_correct_arrivals_total = []
		predicted_vs_correct_arrivals_hour = None
		for day in self.test_days:
			self.last_time_updated = -1
			data, labels, self.station_status = self.load_day(day=day)
			self.station_status_updates = load_station_status_updates(day=day)
			predicted_vs_correct_arrivals_day = []
			hour = -2
			for i, data_point in enumerate(data):
				while (int)(self.last_time_updated) / 60 > hour:
					hour += 1
					predicted_vs_correct_arrivals_hour = {key: [0, 0] for key in self.station_coordinates_or_clusters.keys()}
					predicted_vs_correct_arrivals_day.append(predicted_vs_correct_arrivals_hour)
				pred_label = self.get_prediction(data_point=data_point)
				if pred_label[0] == -1 and pred_label[1] == -1:
					hits_misses_nonguess[2] += 1
				else:
					correct_label = labels[i]
					#error = 0.0
					#for i in range(2):
					#	error += (correct_label[i] - pred_label[i]) ** 2
					error = np.linalg.norm(correct_label-pred_label)
					if error < self.error_threshold:
						hits_misses_nonguess[0] += 1
					else:
						if error > 2:
							print(error, correct_label, pred_label)
						hits_misses_nonguess[1] += 1
					errors.append(error)
					closest_vacant_station = self.find_closest_vacant_station(pred_label)
					correct_station = self.inv_station_coordinates[correct_label[0]][correct_label[1]]
					predicted_vs_correct_arrivals_hour[closest_vacant_station][0] += 1
					predicted_vs_correct_arrivals_hour[correct_station][1] += 1
					"""print(closest_vacant_station)
					print(correct_station)
					print(predicted_vs_correct_arrivals_hour)
					print(hour)
					print(predicted_vs_correct_arrivals_day[hour])"""
					"""if (self.station_status[correct_label] == 0 or self.station_status[pred_label] == 0):
						print("predicted:", pred_label, '   correct:', correct_label,
							  '\nstatus predicted:', self.station_status[pred_label],
							  'status correct:', self.station_status[correct_label])"""

			predicted_vs_correct_arrivals_total.append(predicted_vs_correct_arrivals_day)
		print("Hits vs misses:", hits_misses_nonguess)
		total_guesses = 1.0 * (hits_misses_nonguess[0] + hits_misses_nonguess[1])
		total_guesses += 0.001
		hit_percentage = hits_misses_nonguess[0] / (total_guesses)
		average_error = np.sum(errors) / total_guesses
		median_error = np.mean(errors)
		if total_guesses != 0:
			print("Hit percentage: %.4f" % hit_percentage)
			print("Average error: %.3f" % average_error)
		else:
			print("Didn't guess any")
		return hits_misses_nonguess, hit_percentage, average_error, errors, median_error

	def get_prediction(self, data_point):
		# update station status
		self.last_time_updated = self.update_station_status(data_point)
		# if we can get user prediction:
		user_ID = data_point[-1]
		# print(user_ID, self.predictors.keys())
		if user_ID in self.predictors.keys():
			#print('userID: ', user_ID, 'normalized trip userID: ', data_point[-1])
			pred_label = self.predictors[user_ID].get_prediction(case=data_point,
																	station_status=self.station_status)
			return pred_label
		# else: say we dont want to guess
		if (int)(user_ID)*1.0 != user_ID:
			print("Line 245 in simulator.py")
			print(user_ID)
		return [-1, -1]

	def update_station_status(self, data_point):
		new_time = get_minutes_after_midnight_normalized(data_point)
		if (new_time != self.last_time_updated):
			if self.last_time_updated in self.station_status_updates.keys():
				updates = self.station_status_updates[self.last_time_updated]
				for update in updates:
					#station_number = update[0]
					self.station_status[update[0]] = update[1]
			"""for station in self.station_status.keys():
				available_slots = self.station_status[station][0]
				if available_slots != 0:
					print(station, self.station_status[station])
			"""
		return new_time

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
			# days_matrix = [self.split_date_time_object(date[0], remove_time=True) for date in query_result]
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
			if (number < self.training_size_proportion):
				training_days.append(day)
				counters[0] += 1
			elif (number < self.training_size_proportion + self.test_size_proportion):
				test_days.append(day)
				counters[1] += 1
			else:
				validation_days.append(day)
				counters[2] += 1
		self.training_days = np.array(training_days)
		self.test_days = np.array(test_days)
		print(self.test_days[0])
		self.validation_days = np.array(validation_days)

	# print("Counters: ", counters)

	def run(self):
		self.split_days_into_training_test_vaildation()
		percentage_of_users = (int)(len(self.users) * self.fraction_of_users)
		self.users = self.users[:percentage_of_users]
		self.connect_all_users_to_predictors(users=self.users)
		if self.min_cluster_size > 1:
			station_cluster = {}
			for key in self.station_coordinates_or_clusters.keys():
				value = self.station_coordinates_or_clusters[key]
				if not value in station_cluster.keys():
					station_cluster[value] = []
				station_cluster[value].append(key)
			self.station_coordinates_or_clusters = station_cluster
		hits_misses_nonguesses, hit_percentage, average_error, errors, mean_error = self.test_by_user()
		write_results_to_file(simulator=self,
							  hits_misses_nonguesses=hits_misses_nonguesses,
							  hit_percentage=hit_percentage,
							  average_error=average_error,
							  error_threshold=self.error_threshold,
							  mean_error=mean_error)
		graphics.plot_histogram(data=[errors], title='Error plot', xlabel='Error', xscale='log', save=True)


def get_day_number_out_of_trip(trip, min_cluster_size):
	year_week_weekday = datamanager.get_year_week_week_day(trip=trip, min_cluster_size=min_cluster_size)
	year = year_week_weekday[0]
	week = year_week_weekday[1]
	weekday = year_week_weekday[2]
	#print('daynumber:', year, week, weekday)
	day_number = (year-2016)*365 + week*7 + weekday
	return (day_number)

def get_minutes_after_midnight_normalized(data_point):
	return (data_point[-2])


def load_all_station_coordinates(min_cluster_size=0):
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
	if min_cluster_size > 0:
		print('clustering at line 364 in simulator.py')
		clusterer = datamanager.hdb_cluster(all_coordinates)
		cluster_labels = clusterer.fit(all_coordinates)
		labels = clusterer.fit_predict(all_coordinates)
		station_cluster = {}
		counter = max(labels) +1
		for i, station_number in enumerate(all_stations):
			if labels[i] == -1:
				station_cluster[station_number] = counter
				counter += 1
			else:
				station_cluster[station_number] = labels[i]
		#print(all_coordinates[100000000])
		for value in station_cluster.values():
			pass
			#print(value)
		return station_cluster
	all_coordinates = datamanager.normalize_data(all_coordinates)
	"""print('Max utm0: '+str(max(all_coordinates[:,0])))
	print('Min utm0: '+str(min(all_coordinates[:,0])))
	print('Max utm1: '+str(max(all_coordinates[:,1])))
	print('Min utm1: '+str(min(all_coordinates[:,1])))
	print('argmax: '+str(np.argmax(all_coordinates)))
	print('argmin: '+str(np.argmin(all_coordinates)))"""
	for i, station in enumerate(all_stations):
		utm_coordinates = all_coordinates[i]
		station_coordinates[station] = utm_coordinates
	return station_coordinates


def load_all_users():
	mypath = 'data/trips_by_user/'
	onlyfiles = [f.split('_')[1].split('.')[0] for f in listdir(mypath) if isfile(join(mypath, f))]
	return onlyfiles[1:]


def load_station_status_updates(day):
	updates = {}
	path = 'data/station_availability/day_'
	filename = path + datamanager.convert_date_to_string_year_month_day(day) + '.csv'
	with open(filename, 'r') as file:
		rows = file.readlines()
		for row in rows:
			row_split = row.split(',')
			date_string = row_split[6]
			date = datamanager.convert_string_to_date(date_string, delimiter='-', year_index=0, day_index=2)
			minutes_after_midnight = date.hour * 60 + date.minute
			station_number = (int)(row_split[7])
			online = (row_split[1] == 'true')
			if (online):
				available_bikes = (int)(row_split[4])
				available_slots = (int)(row_split[5])
			else:
				available_bikes, available_slots = 0, 0
			if minutes_after_midnight not in updates.keys():
				updates[minutes_after_midnight] = []
			info = [station_number, available_slots]
			updates[minutes_after_midnight].append(info)
	return updates

def plot_day(prediction_vs_correct, day_key, station_key, cluster=False):
	data = prediction_vs_correct[day_key][station_key]
	y1 = []
	y2 = []
	for minute in data.keys():
		hour = minute/60
		predicted = data[minute][0]
		correct = data[minute][1]
		#print(predicted, correct, hour)
		for i in range(predicted):
			y1.append(hour)
		for i in range(correct):
			y2.append(hour)
	day_key = (int)(day_key)
	year, week, week_day = datamanager.get_year_week_weekday_from_day_number(day_key)
	title = 'Demand_for_year:'+str(year)+'_week:'+str(week)+'_weekday:'+str(week_day)
	if cluster:
		title += '\nFor cluster:_'+str(station_key)
	'+\nstation: '+str(station_key)
	graphics.plot_histogram(data=[y1, y2], number_of_bins=24, title=title,
							xlabel='Hour', ylabel='Number of bikes', show=False, save=True,
	                        path='plots/day_plot/')

def write_results_to_file(simulator, hits_misses_nonguesses, hit_percentage, average_error, error_threshold,
						  mean_error):
	filename = simulator.predictor_func
	path = "results/"
	total_filename = path+filename+'.txt'
	if not (isfile(total_filename)):
		try:
			file = open(total_filename, 'r')
		except IOError:
			file = open(total_filename, 'w')
	with open(total_filename, 'a') as f:
		text = "======================================================="
		text += '\nDatetime: ' + str(datetime.datetime.now())
		text += '\nTrainingSize ' + str(simulator.training_size_proportion)
		text += '\nTestSize ' + str(simulator.test_size_proportion)
		text += '\nFractionOfUsers ' + str(simulator.fraction_of_users)
		text += '\nNormalizeData ' + str(simulator.normalize_data)
		text += '\nCutOffPointDataAmount ' + str(simulator.cut_off_point_data_amount)
		text += '\nErrorThreshold ' + str(error_threshold)
		text += "\n"
		text += '\nHitsMissesNonguesses '
		for value in hits_misses_nonguesses:
			text += str(value) + ", "
		text = text[:-2]
		text += '\nHitPercentage ' + str(hit_percentage)
		text += '\nAverageError ' + str(average_error)
		text += '\nMeanError ' + str(mean_error)
		text += "\n"
		f.write(text)