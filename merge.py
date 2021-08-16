import re
lnc= open("list_lncrna_GC_CPAT_filter.txt","r", encoding ="utf8", errors="ignore")
while True:
   code=lnc.readline()
   if not code:
       break
   
   lncRNA= code.rstrip().split("\t")
   ens=lncRNA[0]
   
   with open("uniq_sorted_di_ntd.csv","r", encoding ="utf8", errors="ignore") as dint:
       while True:
           noncoding=dint.readline()
           if not noncoding:
               break
           noncoding_RNA= noncoding.rstrip().split("\t")
           
           if ens==noncoding_RNA[0]:
               
               merge=lncRNA+"\t"+noncoding_RNA
               
               print(merge)
           
           
   
   
   
