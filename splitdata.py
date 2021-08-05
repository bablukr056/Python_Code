import re
fo = open("gencode.v38.long_noncoding_RNAs.gtf","r", encoding ="utf8", errors="ignore")
while True:
   line = fo.readline()
   if not line:
      break
   res = re.split('\t|"|;', line)
   b=res[0]#+"\t"+res[1]#+"\t"+res[3]+"\t"+res[4]#+"\t"+res[9]#+"\t"+res[13]#+"\t"+res[16]
   print(b)
   #print(res)
   
   
   
   
   
   
   
#ar = line.rstrip().split("-")
#b=re.split("_|:",ar)
