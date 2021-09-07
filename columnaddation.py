import re
fo = open("filename.tsv","r", encoding ="utf8", errors="ignore")
while True:
   line = fo.readline()
   if not line:
      break
   line = line.rstrip()
  
   print(line,"\t1")
   

   
