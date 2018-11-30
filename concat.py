import json

combined_list = []
with open('./data.json', 'rb') as f:
	data = json.load(f)
	print(type(data))
	with open('./instagram_crawler/new_data.json', 'rb') as f2:
		data2 = json.load(f2)
		print(type(data2))
		combined_list = data + data2

with open('updated_data.json', 'w', encoding='utf8') as outfile:
	json.dump(combined_list, outfile, ensure_ascii=False, indent=2)

