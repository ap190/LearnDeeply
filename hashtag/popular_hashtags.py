def popular_hashtags():
	"""
	pull out 10k most popular hashtags
	return: dictionary {hashtag: weight}
	"""
	dic = {}
	with open('./hashtags.txt', 'r', encoding = 'utf8') as f:
		readed = f.read()
		hashtags = readed.split()
		num_hashtags = len(hashtags)
		for i in range(num_hashtags):
			dic[hashtags[i]] = num_hashtags
			num_hashtags -= 1

	return dic
# print hashtag and corresponding weight
# hashtags = popular_hashtags()
# for key, val in hashtags.items():
# 	print(key, val)
