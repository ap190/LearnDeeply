#!/usr/bin/env python3
"""
	This code is used to compute a histogram for the Like to Follower Ratio (LFR)
	for all the posts in our dataset. 
"""
import json
import pandas

def get_LFR_for_user(user_data):
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


def get_LFR_for_all_users():
	"""
	Gets
	"""
	with open("../data.json")  as file:
		data = json.load(file)
		lfr_for_all_posts = []
		for user in data:
			lfr_for_all_posts += get_LFR_for_user(user)
		print(pandas.qcut(lfr_for_all_posts, q=100))
		


if __name__ == "__main__":
    get_LFR_for_all_users()