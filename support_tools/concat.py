import json

"""
Script to create new data json file from old json files.
"""
combined_list = []
with open('./data.json', 'rb') as f:
	data = json.load(f)
	print(type(data))
	with open('./instagram_crawler/new_data.json', 'rb') as f2:
		data2 = json.load(f2)
		with open('./instagram_crawler/new_data2.json', 'rb') as f3:
			data3 = json.load(f3)
			combined_list = data + data2 + data3

with open('updated_data.json', 'w', encoding='utf8') as outfile:
	json.dump(combined_list, outfile, ensure_ascii=False, indent=2)

