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
	                      normalize_data=extracted_config.normalize_data,
	                      closeness_cutoff=extracted_config.cut_off_closeness_measure)
	simulator.run()