#!/usr/bin/python
import re
def printFeat( seq ):
   lst = list(seq.upper())
   ln = len(lst)
   for i in range(0,ln-2,1):
      for j in range(0,dln,1):
         if (lst[i]+lst[i+1]+lst[i+2] == diNucl[j]):
            dict[diNucl[j]] += 1
   for j in range(0,dln,1):
      #print(",",(dict[diNucl[j]]/ln), end="")
      print(",%0.2f" % (dict[diNucl[j]]/ln), end="")
   print()
   return;
diNucl = ['AAA','ATA','AGA','ACA','TAA','TTA','TGA','TCA','GAA','GTA','GGA','GCA','CAA','CTA','CGA','CCA']
dict = {}
seq = ""
bool=0;
dln = len(diNucl)
for j in range(0,dln,1):
   dict[diNucl[j]]=0
print("SeqID", end="")
for j in range(0,dln,1):
   print(",",diNucl[j], end="")
print()
fo = open("list_ENST_lncRNA_mortazavi_human.fasta", "r")
while True:
   line = fo.readline()
   if (re.search("^>", line)):
      if(bool):
         print(id,end="")
         printFeat(seq)
      id = line.rstrip()
      id = id.strip(">")
      seq = ""
      for j in range(0,dln,1):
         dict[diNucl[j]]=0
   if (re.search("^\w+", line)):
      seq = seq + line.rstrip()
      bool=1
   # check if line is not empty
   if not line:
      break
fo.close()
print(id,end="")
printFeat(seq)
