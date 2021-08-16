import re


fo = open("list_lncrna_mortazavi_human_GC_CPAT.txt","r", encoding ="utf8", errors="ignore")
while True:
   line = fo.readline()
   if not line:
      break
   ar = line.rstrip().split("\t")
   if(re.search("GC%",ar[0])and re.match("Fickett_score",ar[4])):
         sample =ar[3]+"\t"+ar[4]
         print(sample)
fo.close()
