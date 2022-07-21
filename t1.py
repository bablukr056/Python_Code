from operator import length_hint
import re 

header = None
length = 0
with open('nextprot_all.peff') as fasta:   #reading the input file
    for line in fasta:
        line = line.rstrip()
        if line.startswith('>'):          #start with > symbole
            if header is not None:
                print(header, length)
                length = 0
            line=re.split(":| ",line) #split based on : and space
            header=line[1]   #print  accession id which stored in list
        else:
            length =length+ len(line)
if length:
    print(header, length)                