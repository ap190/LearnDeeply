import json
import numpy as np
import dateutil.parser as dateparser
from bins import *
from keras import *
from common import *
from math import *

path = './data.json'
def preprocessing(path = path):
	# [num_posts, num_following, num_followers, num_tags, description_length]
	inputs, outputs = [], []

	with open(path, 'rb') as f:
		loaded_json = json.load(f)
		#BIN = bins(100)
		#interval, sequence, bin_edges = BIN.even_bins()
		for user in loaded_json:
			for post in user['images']:
				num_posts = user['posts']
				num_following = user['following']
				num_followers = user['followers']
				date = dateparser.parse(post['date'])			
				p = [num_interp(num_following), 
					 num_interp(num_followers),
					 num_interp(user['posts']),
					 len(post['tags']),
					 len(post['description']),
					 len(post['mentions']),
					 date.weekday(),
					 date.hour]
				num_likes = post['likes']
				if type(num_likes) is str:
					num_likes = int(num_likes.replace(",", ""))
				if num_likes <= 0 or num_likes > 200000:
					continue
				"""
				if num_likes < 10870:
					label = BIN.bin_classification(num_likes)
					inputs.append(p)
					outputs.append(label)
				"""
				inputs.append(p)
				outputs.append(np.log(num_likes))
	print(len(inputs))
	print(len(outputs))
	#print(inputs)
	#print(outputs)
	#stop
	return np.asarray(inputs), np.asarray(outputs)


class meta_cnn:
	def __init__(self, input_data, input_labels, hidden_layers=0, hidden_sizes=[]):
		np.random.shuffle(input_data)
		np.random.shuffle(input_labels)
		self.data = input_data
		self.labels = input_labels
		self.input_data = self.data[2000:,:]
		self.input_labels = self.labels[2000:] #np.eye(np.max(input_labels)+1)[input_labels]
		self.test_data = self.data[:2000,:]
		self.test_labels = self.labels[:2000]
		self.num_classes = 1 #len(self.input_labels[0, :])
		self.hidden_layers = hidden_layers
		self.hidden_sizes = hidden_sizes
		self.batch_size = 30

		self.model = Sequential()
		self.construct_model()

	def construct_model(self):
		if not self.hidden_layers:
			self.model.add(layers.Dense(self.num_classes, input_shape=(len(self.input_data[0], ))))
			# self.model.add(layers.Dense(np.int32(np.max(self.input_labels)), input_dim=len(self.input_data[0]), activation='sigmoid'))
		else:
			# self.model.add(layers.Dense(self.hidden_sizes[0], input_dim=len(self.input_data[0]), activation='relu'))
			self.model.add(layers.Dense(self.hidden_sizes[0], input_shape=(len(self.input_data[0]), ), activation='relu'))
			for i in range(1, self.hidden_layers):
				self.model.add(layers.Dense(self.hidden_sizes[i], activation='relu'))

			self.model.add(layers.Dense(self.num_classes))

		optimizer = optimizers.Adam(lr=0.01)
		self.model.compile(loss='mape', optimizer=optimizer, metrics=["mae", "mse"])

	def train_model(self):
		# print(self.input_data[:50])
		# print(self.input_labels[:50])
		# stop
		self.model.fit(self.input_data, self.input_labels, self.batch_size, epochs=5, verbose=1, validation_split=0.1)

	def test_model(self):
		self.prediction = self.model.predict(self.test_data)
		return self.test_data, self.prediction

	def test_error(self):
		return np.mean((np.exp(self.prediction) - self.test_labels))

	def item_error(self):
		for i in range(len(self.test_labels)):
			result = [exp(self.prediction[i]), exp(self.test_labels[i]), exp(self.prediction[i]) - exp(self.test_labels[i])]
			print(result)

inputs, outputs = preprocessing()
MDL = meta_cnn(inputs, outputs, 3, [100, 100, 100])
MDL.train_model()
test_data, prediction = MDL.test_model()
print(prediction.shape)
print(np.unique(prediction))
print(MDL.test_error())
print(MDL.item_error())



