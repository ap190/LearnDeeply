import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

class bins:
	def __init__(self, bin_nums = 10, path = "./profiles"):
		self.bin_nums = bin_nums
		self.path = path
		self.counted, self.likes_array = self.count_elements()

	def count_elements(self):
		"""
		Count the elements, return counted, likes_array.
		Input: Null
		Output: counted[dictionary{num_likes: frequency}], likes_array[sequence]
		"""
		likes_array = []
		for root, dirs, files in os.walk(self.path):
			for file in files:
				with open(self.path + '/' + file, 'rb') as f:
					if not file.endswith('.json'):
						continue
					loaded_json = json.load(f)
					for post in loaded_json['posts']:
						num_likes = post['numberLikes']
						if type(num_likes) is str:
							num_likes = int(num_likes.replace(",", ""))
						likes_array.append(num_likes)
		counted = Counter(likes_array)
		return counted, likes_array

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

# BIN = bins(bin_nums = 100)
# BIN.visualize_histogram()
