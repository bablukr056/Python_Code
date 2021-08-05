#extract data

import re
human = open("gencode.v38.pc_transcripts.fa.gz","r", encoding ="utf8", errors="ignore")
while True:
   code=human.readline()
   if not code:
       break
   coding= code.rstrip().split("\t")
   feature=coding[0]#+"\t"+coding[2]+"\t"+coding[3]+"\t"+coding[4]
   print(feature)
   
   
