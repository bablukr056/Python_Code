import re
fo = open("orfhumancoding.tsv","r", encoding ="utf8", errors="ignore")
while True:
   line = fo.readline()
   if not line:
      break
   line = line.rstrip()
   #line["x"]=1
   #res = re.split(' |\t', line)
   #b=res[0]+"\t"+res[1]+"\t"+res[2]+"\t"+res[3]+"\t"+res[4]+"\t"+res[5]#+"\t"+res[6]
   print(line,"\t1")
   

   
