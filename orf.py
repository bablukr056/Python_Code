import re


fo = open("gencodehumanlncrna.csv","r", encoding ="utf8", errors="ignore")
while True:
   line = fo.readline()
   if not line:
      break
   ar = line.rstrip().split("\t")
   
   orf=ar[0]+"\t"+ar[1]+"\t"+ar[2]+"\t"+ar[3]+"\t"+ar[4]+"\t"+ar[8]
   
   print(orf)
