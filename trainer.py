
import numpy as np
from scipy import spatial
from datamanager import DataManager
from copy import deepcopy


class Predictors:

	def __init__(self, trip_data, test_trip_data, all_trips):
		self.all_trips = all_trips
		self.trip_data = trip_data
		self.test_trip_data = test_trip_data

	def find_best_matching_trip(self, trip):
		tree = spatial.KDTree(self.test_trip_data)
		best_trip = tree.query(trip)
		distance = best_trip[0]
		best_trip = best_trip[1]
		return best_trip

	def prediction_of_end_station_by_find_best_matching_trip(self, trip):
		best_trip = self.find_best_matching_trip(trip=trip)
		end_station = self.all_trips[best_trip][1]
		return end_station


if __name__ == '__main__':
	data_manager = DataManager()
	data_manager.set_normalized_trips_for_user()
	predictors = Predictors(trip_data=data_manager.all_normalized_trips,
	                        test_trip_data=data_manager.all_normalized_trips_without_end_station,
	                        all_trips=data_manager.all_trips)
	trip = predictors.test_trip_data[-1]
	best_matching_trip = predictors.prediction_of_end_station_by_find_best_matching_trip(trip)
	print(trip, best_matching_trip)