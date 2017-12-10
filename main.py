from datamanager import DataManager
from trainer import Predictor

#########################################
# export GOOGLE_APPLICATION_CREDENTIALS=/Users/johan/PycharmProjects/Fordypningsprosjekt/'UIP students-8ffa90ab95f7.json'
#########################################

if __name__ == '__main__':
	print("main.py main is running")
	data_manager = DataManager(user_ID=5701)
	data_manager.set_normalized_trips_for_user()
	data_manager.set_training_test_validation_trips()
	print("run predictors")
	predictor = Predictor(training_data=data_manager.training_trips,
	                       test_data=data_manager.test_trips,
	                       validation_data=data_manager.validation_trips,
	                       training_labels=data_manager.training_labels,
	                       test_labels=data_manager.test_labels,
	                       validation_labels=data_manager.validation_labels)
	trip = predictor.test_data[0]
	best_matching_trip, best_matching_end_station \
		= predictor.get_prediction(case=trip)
	print(best_matching_trip, best_matching_end_station)
	print(trip, predictor.test_labels[0])

	predictor.run_all_test_data()

	print("executed main in main.py")