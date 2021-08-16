
data = data2 = ""

with open('1_added_coding.tsv') as fp:
	data = fp.read()

with open('0_added_lncrna.tsv') as fp:
	data2 = fp.read()

data += "\n"
data += data2

with open ('morta_svm_combined.tsv', 'w') as fp:
	fp.write(data)
