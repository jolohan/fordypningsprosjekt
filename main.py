from datamanager import DataManager
from predictor import Predictor
from extract_config import ExtractConfig
from simulator import Simulator

#########################################
# export GOOGLE_APPLICATION_CREDENTIALS=/Users/johan/PycharmProjects/Fordypningsprosjekt/'UIP students-8ffa90ab95f7.json'
#########################################

if __name__ == '__main__':
	extracted_config = ExtractConfig()
	simulator = Simulator(training_size_proportion=extracted_config.training_size,
	                      test_size_proportion=extracted_config.test_size,
	                      fraction_of_users=extracted_config.fraction_of_users,
	                      predictor_func=extracted_config.predictor_func,
	                      normalize_data=extracted_config.normalize_data)
	simulator.run()
















def old_main():
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