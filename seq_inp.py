import re
fo = open("cntr_2.tsv","r")

while True:
    line  = fo.readline()
    if not line:
        break
    line=re.split('\t',line)
    print(line[0]+'\t'+line[2]+'\t'+line[3])
