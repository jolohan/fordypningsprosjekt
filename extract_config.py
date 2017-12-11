








class ExtractConfig():

	def __init__(self, which_config='test_1'):
		self.load_config(which_config)

	def load_config(self, which_config):
		self.parameters = {}
		filename = 'config/' + which_config + '.txt'
		with open(filename, 'r') as file:
			for line in file:
				listed_specs = line.split(" ")

				self.parameters[listed_specs[0]] = [item.strip() for item in listed_specs[1:]]

		print(self.parameters)
		self.training_size = (float)(self.parameters['TrainingSize'][0])
		self.test_size = (float)(self.parameters['TestSize'][0])
		self.fraction_of_users = (float)(self.parameters['FractionOfUsers'][0])
		self.predictor_func = self.parameters['Predictor'][0]