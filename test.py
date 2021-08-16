#!/usr/bin/python

import re

def printFeat( seq ):
   lst = list(seq.upper())
   ln = len(lst)
   for i in range(0,ln-1,1):
      for j in range(0,dln,1):
         if (lst[i]+lst[i+1] == diNucl[j]):
            dict[diNucl[j]] += 1
   print(id,dict)
   return;



diNucl = ['AA','AT','AG','AC','TA','TT','TG','TC','GA','GT','GG','GC','CA','CT','CG','CC']
dict = {}
seq = ""
bool=0;
dln = len(diNucl)
for j in range(0,dln,1):
   dict[diNucl[j]]=0
fo = open("list_lncrna_mortazavi_human.fasta", "r")
while True:
   line = fo.readline()
   if (re.search("^>", line)):
      if(bool):
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
printFeat(seq)
