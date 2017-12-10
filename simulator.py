from trainer import Predictor
import datamanager
from os import listdir
from os.path import isfile, join

class Simulator():



	def __init__(self):
		self.predictors = {}
		users = get_all_users()
		percentage_of_users = (int)(len(users)*0.001)
		self.connect_all_users_to_predictors(users=users[:percentage_of_users])

	def connect_all_users_to_predictors(self, users):
		for i, user in enumerate(users):
			if (i%100 == 0):
				print(i, "/", len(users))
			self.connect_single_user_to_predictor(user_ID=user)

	def connect_single_user_to_predictor(self, user_ID):
		data_manager = datamanager.DataManager(user_ID=user_ID)
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
		formatted_csv_result = datamanager.format_csv_result(filename=(path+day))
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

	def run(self):
		self.simulate_day()

def get_all_users():
	mypath = 'data/trips_by_user/'
	onlyfiles = [f.split('_')[1].split('.')[0] for f in listdir(mypath) if isfile(join(mypath, f))]
	return onlyfiles[1:]

if __name__ == '__main__':
	simulator = Simulator()
	simulator.run()