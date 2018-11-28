# common functions for processing data
# sometimes the numbers are interpretted as strings, sometimes they contain an 'm' for million,
# either way, it's a pain to deal with so in every single individual function, so here are common
# functions for dealing with them


# convert numbers that are represented as strings into actual integer numbers
def num_interp(number):
	# if the input 'number' is a string, we need to edit
	if type(number) is str:
		# replace any commas that might exist
		number = number.replace(',', '')

		# if the number is in the millions, there will be an 'm' instead
		if number.endswith('m'): 
			number = int(float(number[:-1]) * 1000000)
		# if number is in the thousands, there will be a 'k' instead
		elif number.endswith('k'):
			number = int(float(number[:-1]) * 1000)

		return number
	else:
		return number