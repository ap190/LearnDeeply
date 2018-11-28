import json
import numpy as np
from bins import *
from keras import *

path = './data.json'
def preprocessing(path = path):
	# [num_posts, num_following, num_followers, num_tags, description_length]
	inputs, outputs = [], []

	with open(path, 'rb') as f:
		loaded_json = json.load(f)
		BIN = bins(100)
		interval, sequence, bin_edges = BIN.even_bins()
		for user in loaded_json:
			for post in user['images']:
				num_posts = user['posts']
				num_following = user['following']
				num_followers = user['followers']
				num_tags = len(post['tags'])
				description_length = len(post['description'])				
				p = [num_interp(num_posts), 
					 num_interp(num_following), 
					 num_interp(num_followers), 
					 num_tags, 
					 description_length]
				num_likes = post['likes']
				if type(num_likes) is str:
					num_likes = int(num_likes.replace(",", ""))
				label = BIN.bin_classification(num_likes)
				inputs.append(p)
				outputs.append(label)

	return np.asarray(inputs), outputs


class meta_cnn:
	def __init__(self, input_data, input_labels, hidden_layers=0, hidden_sizes=[]):
		self.input_data = input_data
		self.input_labels = np.eye(np.max(input_labels)+1)[input_labels]
		self.num_classes = len(self.input_labels[0, :])
		self.hidden_layers = hidden_layers
		self.hidden_sizes = hidden_sizes
		self.batch_size = 100

		self.model = Sequential()
		self.construct_model()

	def construct_model(self):
		if not self.hidden_layers:
			self.model.add(layers.Dense(self.num_classes, input_shape=(len(self.input_data[0]), ), activation='sigmoid'))
			# self.model.add(layers.Dense(np.int32(np.max(self.input_labels)), input_dim=len(self.input_data[0]), activation='sigmoid'))
		else:
			# self.model.add(layers.Dense(self.hidden_sizes[0], input_dim=len(self.input_data[0]), activation='relu'))
			self.model.add(layers.Dense(self.hidden_sizes[0], input_shape=(len(self.input_data[0]), ), activation='relu'))
			for i in range(1, self.hidden_layers):
				self.model.add(layers.Dense(self.hidden_sizes[i], activation='relu'))

			self.model.add(layers.Dense(self.num_classes, activation='sigmoid'))

		self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	def train_model(self):
		self.model.fit(self.input_data, self.input_labels, batch_size=10, epochs=1, verbose=1)

inputs, outputs = preprocessing()
MDL = meta_cnn(inputs, outputs, 1, [200000])
MDL.train_model()



