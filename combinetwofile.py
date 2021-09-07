
data = data2 = ""

with open('file_1.tsv') as fp:
	data = fp.read()

with open('file_2.tsv') as fp:
	data2 = fp.read()

data += "\n"
data += data2

with open ('combinefile_1_2', 'w') as fp:
	fp.write(data)
