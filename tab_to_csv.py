import csv

with open("drop_morta_svm.tsv", 'r') as infile, open("bablu_csvfile.csv", 'w') as outfile:
     stripped = (line.strip() for line in infile)
     lines = (line.split(",") for line in stripped if line)
     writer = csv.writer(outfile)
     writer.writerows(lines)
