with open('alias_4.txt', 'r', encoding='utf16') as f:
	loaded = f.read().split()
	half = len(loaded)//2
	with open('alias_4_1.txt', 'w', encoding='utf16') as w:
		for i in range(half):
			w.write(loaded[i] + ' ')
	with open('alias_4_2.txt', 'w', encoding='utf16') as p:
		for i in range(half, len(loaded)):
			p.write(loaded[i] + ' ')
