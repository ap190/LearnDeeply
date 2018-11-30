import json
import numpy as np
import dateutil.parser as dateparser
from bins import *
from keras import *
from common import *
from math import *
from hashtag.popular_hashtags import *

path = './data.json'

hash_weights = popular_hashtags()

def preprocessing(path = path):
	"""
	return: inputs[taining_data], outputs[training_labels]
	inputs: [num_following, num_followers, num_posts, num_tags, 
			description_length, num_mentions, week_of_the_day, 
			hour_of_the_day, tag_weight, standard_variation]
	outputs: [log_num_likes]
	type: nparray
	"""
	inputs, outputs = [], []

	print('Loading data...')

	with open(path, 'rb') as f:
		max_num_following = 0
		max_num_followers = 0
		max_num_posts = 0
		max_num_tags = 0
		max_description_length = 0
		max_num_mentions = 0
		max_week_of_the_day = 6
		max_hour_of_the_day = 23
		max_tag_weight = 0
		max_num_likes = 0
		loaded_json = json.load(f)
		for user in loaded_json:
			for post in user['images']:
				num_likes = post['likes']
				if type(num_likes) is str:
					num_likes = int(num_likes.replace(",", ""))
				if num_likes <= 0 or num_likes > 500000:
					continue
				max_num_likes = max(max_num_likes, num_likes)
				max_num_following = max(max_num_following, num_interp(user['following']))
				max_num_followers = max(max_num_followers, num_interp(user['followers']))
				max_num_posts = max(max_num_posts, num_interp(user['posts']))
				max_num_tags = max(max_num_tags, len(post['tags']))
				max_description_length = max(max_description_length, len(post['description']))
				max_num_mentions = max(max_num_mentions, len(post['mentions']))
				tag_weight = 0
				for tag in post['tags']:
					if tag[1:] in hash_weights:
						tag_weight += hash_weights[tag[1:]]
				max_tag_weight = max(max_tag_weight, tag_weight)

	print(max_num_following, max_num_followers, max_num_posts, max_num_tags, 
		max_description_length, max_num_mentions, max_week_of_the_day, 
		max_hour_of_the_day, max_tag_weight, max_num_likes)

	with open(path, 'rb') as f:
		loaded_json = json.load(f)
		for user in loaded_json:
			likes_per_post = []
			data_without_variance = []
			for post in user['images']:
				num_posts = user['posts']
				num_following = user['following']
				num_followers = user['followers']
				date = dateparser.parse(post['date'])
				tag_weight = 0
				for tag in post['tags']:
					if tag[1:] in hash_weights:
						tag_weight += hash_weights[tag[1:]]		
				p = [num_interp(num_following)/max_num_following, 
					 num_interp(num_followers)/max_num_followers,
					 num_interp(user['posts'])/max_num_posts,
					 len(post['tags'])/max_num_posts,
					 len(post['description'])/max_description_length,
					 len(post['mentions'])/max_num_mentions,
					 date.weekday()/max_week_of_the_day,
					 date.hour/max_hour_of_the_day,
					 tag_weight/max_tag_weight]
				num_likes = post['likes']
				if type(num_likes) is str:
					num_likes = int(num_likes.replace(",", ""))
				if num_likes <= 0 or num_likes > 500000:
					continue
				likes_per_post.append(log(num_likes))
				outputs.append(log(num_likes))
				data_without_variance.append(p)

			variance = np.std(likes_per_post)
			for p in data_without_variance:
				p.append(variance)
				inputs.append(p)

	inputs = np.asarray(inputs)
	outputs = np.asarray(outputs)

	np.random.shuffle(inputs)
	np.random.shuffle(outputs)

	print('Data loaded! :)')
	print('number of inputs: %d' % (len(inputs)))
	print('number of outputs: %d' % (len(outputs)))

	# print to see 20 from inputs and outputs
	# print(inputs[:20])
	# print(outputs[:20])
	# stop

	return inputs, outputs


class meta_cnn:
	def __init__(self, data, labels, hidden_layers=0, hidden_sizes=[]):

		self.data = data
		self.labels = labels

		# splitting data for training and testing
		self.input_data = self.data[2000:,:]
		self.input_labels = self.labels[2000:] #np.eye(np.max(input_labels)+1)[input_labels]
		self.test_data = self.data[:2000,:]
		self.test_labels = self.labels[:2000]

		self.num_classes = 1 #len(self.input_labels[0, :])
		self.hidden_layers = hidden_layers
		self.hidden_sizes = hidden_sizes
		self.batch_size = 30
		self.epochs = 5

		self.model = Sequential()
		self.construct_model()

	def construct_model(self):
		"""
		setting up the feed-foward model
		"""
		if not self.hidden_layers:
			self.model.add(layers.Dense(self.num_classes, input_shape=(len(self.input_data[0], ))))
		else:
			self.model.add(layers.Dense(self.hidden_sizes[0], input_shape=(len(self.input_data[0]), ), activation='relu'))
			for i in range(1, self.hidden_layers):
				self.model.add(layers.Dense(self.hidden_sizes[i], activation='relu'))

			self.model.add(layers.Dense(self.num_classes))

		optimizer = optimizers.Adam(lr=0.001)
		self.model.compile(loss='mae', optimizer=optimizer, metrics=["mae", "mse"])

	def train_model(self):
		"""
		training model
		"""
		self.model.fit(self.input_data, self.input_labels, self.batch_size, self.epochs, verbose=1, validation_split=0.1)

	def test_model(self):
		"""
		testing model on test_data set, print out test error
		return: test_data, prediction
		"""
		self.prediction = self.model.predict(self.test_data)
		self.test_error()
		return self.test_data, self.prediction

	def test_error(self):
		"""
		print out test error
		"""
		print('test_error: %f' % (np.mean(abs(self.prediction - self.test_labels))))
		return 

	def train_error(self):
		"""
		print out train error
		"""
		train_prediction = self.model.predict(self.input_data)
		print('train_error: %f' % (np.mean(abs(train_prediction - self.input_labels))))
		return

	def item_error(self):
		"""
		print out some [prediction, real_label, mae]
		"""
		for i in range(len(self.test_labels[:])):
			print("prediction: %.3f | real_label: %.3f | mae: %.3f" % (exp(self.prediction[i]), exp(self.test_labels[i]), abs(exp(self.prediction[i]) - exp(self.test_labels[i]))))
		return

inputs, outputs = preprocessing()
MDL = meta_cnn(inputs, outputs, 3, [200, 200, 100, 100, 100])
MDL.train_model()
test_data, prediction = MDL.test_model()
print('unique predictions on test dataset: %d' % (len(np.unique(prediction))))
MDL.item_error()
MDL.train_error()
MDL.test_error()


