from datamanager import DataManager
from trainer import Predictors

if __name__ == '__main__':
	print("main.py main is running")
	data_manager = DataManager()
	print("inited data manager")
	data_manager.set_normalized_trips_for_user()
	print("run predictors")
	predictors = Predictors(trip_data=data_manager.all_normalized_trips,
	                        test_trip_data=data_manager.all_normalized_trips_without_end_station,
	                        all_trips=data_manager.all_trips)
	trip = predictors.test_trip_data[-1]
	best_matching_trip = predictors.prediction_of_end_station_by_find_best_matching_trip(trip)
	print(trip, best_matching_trip)
	data_manager.set_training_trips()
	print("executed main in main.py")