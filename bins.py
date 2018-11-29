import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

class bins:
	def __init__(self, bin_nums = 10):
		self.bin_nums = bin_nums
		self.likes_array = self.count_elements()
		self.sequence, self.bin_edges = self.numpy_histogram()

	def count_elements(self):
		"""
		Count the elements, return counted, likes_array.
		Input: Null
		Output: counted[dictionary{num_likes: frequency}], likes_array[sequence]
		"""
		likes_array = []
		with open("data.json", 'rb') as f:
			loaded_json = json.load(f)
			for user in loaded_json:
				for post in user['images']:
					num_likes = post['likes']
					if type(num_likes) is str:
						num_likes = int(num_likes.replace(",", ""))
					if num_likes < 10870:
						likes_array.append(num_likes)
		print(len(likes_array))
		counted = Counter(likes_array)
		return likes_array

	def get_LFR_for_user(self, user_data):
		"""
			Returns a list of LFR for a users posts.
		"""
		num_folowers = user_data["followers"]
		if type(num_folowers) is str:
			num_folowers = num_folowers.replace(",", "")
			if "m" in num_folowers:
				num_folowers = num_folowers.replace("m", "")
				num_folowers = float(num_folowers) * 1000000
			elif "k" in num_folowers:
				num_folowers = num_folowers.replace("k", "")
				num_folowers = float(num_folowers) * 1000
			num_folowers = int(num_folowers)

		posts = user_data["images"]
		lfr_for_posts = []
		for post in posts:
			num_likes = post["likes"]
			if type(num_likes) is str:
				num_likes = num_likes.replace(",", "")
				num_likes = int(num_likes)

			lfr_for_posts.append(round(num_likes/num_folowers, 4))
		return lfr_for_posts

	def get_LFR_for_all_users(self):
		"""
		Gets
		"""
		with open("data.json")  as file:
			data = json.load(file)
			lfr_for_all_posts = []
			for user in data:
				lfr_for_all_posts += self.get_LFR_for_user(user)
		return lfr_for_all_posts	

	def ascii_histogram(self):
		"""
		Print frequency histogram in terminal.
		Input: Null
		Output: Null
		"""
		for k in sorted(self.counted):
			print('{0:5d} {1}'.format(k, '+' * self.counted[k]))

	def numpy_histogram(self):
		"""
		Input: bin_nums[number of bins]
		Output: hist[y axies for histogram], bin_edges[x axies for histogram]
		"""
		hist, bin_edges = np.histogram(self.likes_array, bins = self.bin_nums)
		# Show hist and bin_edges
		print(hist)
		print(bin_edges)

		return hist, bin_edges

	def visualize_histogram(self):
		"""
		Visualize the histogram.
		Input: likes_array[sequence], bin_nums[number of bins].
		Output: Null
		"""
		plt.hist(self.likes_array, bins = self.bin_nums)
		plt.xlabel('Bins(number of likes)')
		plt.ylabel('Posts')
		plt.title('Histogram of %d Even Distributed Bins' % (self.bin_nums))
		plt.show()
		return

	def even_bins(self):
		"""
		Create even bins based on sequence and number of bins.
		Input: likes_array[sequence], bin_nums[number of bins]
		Output: interval[number of posts in each bin], sequence[sequence of posts], bin_edges[x axies for histogram]
		"""
		likes_array = sorted(self.likes_array)
		hist, bin_edges = [], []
		interval = len(likes_array)//self.bin_nums
		for i in range(len(likes_array)):
			if i%interval == 0:
				bin_edges.append(likes_array[i])
		sequence = likes_array[:self.bin_nums*interval]
		return interval, sequence, bin_edges

	def bin_classification(self, likes):
		for i in range(1, len(self.bin_edges)):
			if (likes > self.bin_edges[i-1]) and (likes <= self.bin_edges[i]):
				return i-1

		return i-1

# BIN = bins(bin_nums = 100)
# BIN.visualize_histogram()
