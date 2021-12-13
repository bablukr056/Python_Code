#---------------Dinucleotide_frequecny-------------------
import re
def printFeat( seq ):
   lst = list(seq.upper())
   ln = len(lst)
   for i in range(0,ln-1,1):
      for j in range(0,dln,1):
         if (lst[i]+lst[i+1] == diNucl[j]):
            dict[diNucl[j]] += 1
   for j in range(0,dln,1):
      #print(",",(dict[diNucl[j]]/ln), end="")
      print(",%0.2f" % (dict[diNucl[j]]/ln), end="")
   print()
   return;
diNucl = ['AA','AT','AG','AC','TA','TT','TG','TC','GA','GT','GG','GC','CA','CT','CG','CC']
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
fo = open("filename.fasta", "r")
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
#-----------------------------Tri_nt_frquecny-----------------------
import re
def printFeat( seq ):
   lst = list(seq.upper())
   ln = len(lst)
   for i in range(0,ln-2,1):
      for j in range(0,dln,1):
         if (lst[i]+lst[i+1]+lst[i+2] == triNucl[j]):
            dict[triNucl[j]] += 1
   for j in range(0,dln,1):
      #print(",",(dict[triNucl[j]]/ln), end="")
      print(",%0.2f" % (dict[triNucl[j]]/ln), end="")
   print()
   return;

triNucl = ['AAA','AAT','AAG','AAC','ATA','ATT','ATG','ATC','AGA','AGT','AGG','AGC','ACA','ACT','ACG','ACC','TAA','TAT','TAG','TAC','TTA','TTT','TTG','TTC','TGA','TGT','TGG','TGC','TCA','TCT','TCG','TCC','GAA','GAT','GAG','GAC','GTA','GTT','GTG','GTC','GGA','GGT','GGG','GGC','GCA','GCT','GCG','GCC','CAA','CAT','CAG','CAC','CTA','CTT','CTG','CTC','CGA','CGT','CGG','CGC','CCA','CCT','CCG','CCC']
dict = {}
seq = ""
bool=0;
dln = len(triNucl)
for j in range(0,dln,1):
   dict[triNucl[j]]=0

print("SeqID", end="")
for j in range(0,dln,1):
   print(",",triNucl[j], end="")
print()

fo = open("liver_filtered.fasta", "r")
while True:
   line = fo.readline()
   if (re.search("^>", line)):
      if(bool):
         print(id,end="")
         printFeat(seq)
#-----------------------------------Tetra_nt_frquency-----------------------------------
import re
def printFeat( seq ):
   lst = list(seq.upper())
   ln = len(lst)
   for i in range(0,ln-3,1):
      for j in range(0,dln,1):
         if (lst[i]+lst[i+1]+lst[i+2]+lst[i+3] == tetraNucl[j]):
            dict[tetraNucl[j]] += 1
   for j in range(0,dln,1):
      print(",%0.2f" % (dict[tetraNucl[j]]/ln), end="")
   print()
   return;

tetraNucl = ['AAAA','AAAT','AAAG','AAAC','AATA','AATT','AATG','AATC','AAGA','AAGT','AAGG','AAGC','AACA','AACT','AACG','AACC','ATAA','ATAT','ATAG','ATAC','ATTA','ATTT','ATTG','ATTC','ATGA','ATGT','ATGG','ATGC','ATCA','ATCT','ATCG','ATCC','AGAA','AGAT','AGAG','AGAC','AGTA','AGTT','AGTG','AGTC','AGGA','AGGT','AGGG','AGGC','AGCA','AGCT','AGCG','AGCC','ACAA','ACAT','ACAG','ACAC','ACTA','ACTT','ACTG','ACTC','ACGA','ACGT','ACGG','ACGC','ACCA','ACCT','ACCG','ACCC','TAAA','TAAT','TAAG','TAAC','TATA','TATT','TATG','TATC','TAGA','TAGT','TAGG','TAGC','TACA','TACT','TACG','TACC','TTAA','TTAT','TTAG','TTAC','TTTA','TTTT','TTTG','TTTC','TTGA','TTGT','TTGG','TTGC','TTCA','TTCT','TTCG','TTCC','TGAA','TGAT','TGAG','TGAC','TGTA','TGTT','TGTG','TGTC','TGGA','TGGT','TGGG','TGGC','TGCA','TGCT','TGCG','TGCC','TCAA','TCAT','TCAG','TCAC','TCTA','TCTT','TCTG','TCTC','TCGA','TCGT','TCGG','TCGC','TCCA','TCCT','TCCG','TCCC','GAAA','GAAT','GAAG','GAAC','GATA','GATT','GATG','GATC','GAGA','GAGT','GAGG','GAGC','GACA','GACT','GACG','GACC','GTAA','GTAT','GTAG','GTAC','GTTA','GTTT','GTTG','GTTC','GTGA','GTGT','GTGG','GTGC','GTCA','GTCT','GTCG','GTCC','GGAA','GGAT','GGAG','GGAC','GGTA','GGTT','GGTG','GGTC','GGGA','GGGT','GGGG','GGGC','GGCA','GGCT','GGCG','GGCC','GCAA','GCAT','GCAG','GCAC','GCTA','GCTT','GCTG','GCTC','GCGA','GCGT','GCGG','GCGC','GCCA','GCCT','GCCG','GCCC','CAAA','CAAT','CAAG','CAAC','CATA','CATT','CATG','CATC','CAGA','CAGT','CAGG','CAGC','CACA','CACT','CACG','CACC','CTAA','CTAT','CTAG','CTAC','CTTA','CTTT','CTTG','CTTC','CTGA','CTGT','CTGG','CTGC','CTCA','CTCT','CTCG','CTCC','CGAA','CGAT','CGAG','CGAC','CGTA','CGTT','CGTG','CGTC','CGGA','CGGT','CGGG','CGGC','CGCA','CGCT','CGCG','CGCC','CCAA','CCAT','CCAG','CCAC','CCTA','CCTT','CCTG','CCTC','CCGA','CCGT','CCGG','CCGC','CCCA','CCCT','CCCG','CCCC']
dict = {}
seq = ""
bool=0;
dln = len(tetraNucl)
for j in range(0,dln,1):
   dict[tetraNucl[j]]=0

print("SeqID", end="")
for j in range(0,dln,1):
   print(",",tetraNucl[j], end="")
print()

fo = open("Ffile", "r")
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
         dict[tetraNucl[j]]=0
   if (re.search("^\w+", line)):
      seq = seq + line.rstrip()
      bool=1
   if not line:
      break
fo.close()
print(id,end="")
printFeat(seq)
#--------------------------ATGC_count_in_Fasta_file--------------
input_file = open('lkh_liver.fasta', 'r')
output_file = open('nucleotide_counts.tsv','w')
output_file.write('Gene\tA\tC\tG\tT\tLength\tCG%\n')
from Bio import SeqIO
for cur_record in SeqIO.parse(input_file, "fasta") :
    gene_name = cur_record.name
    A_count = cur_record.seq.count('A')
    C_count = cur_record.seq.count('C')
    G_count = cur_record.seq.count('G')
    T_count = cur_record.seq.count('T')
    length = len(cur_record.seq)
    cg_percentage = float(C_count + G_count) / length
    output_line = '%s\t%i\t%i\t%i\t%i\t%i\t%f\n' % \
    (gene_name, A_count, C_count, G_count, T_count, length, cg_percentage)
    output_file.write(output_line)
output_file.close()
input_file.close()
#--------------------------Extract_Sequence_from_Rawfile---------------------------------------
import re
fo = open("filename.fasta","r",encoding = "utf8", errors='ignore')
while True:
    line  = fo.readline()
    if not line:
        break
    v = "N"
    s = ">"
    if(re.search(">",line)):
        id=line.rstrip()
        continue
    if not (v in line):
        print(id+"\n"+line.rstrip())
#-----------------------Hot_code---------------------------------------------------------------
fo = open("filename.fasta","r",encoding = "utf8", errors='ignore')
line  = fo.readline()
v = "N"
s = ">"
n_a,n_t,n_g,n_c = "A","T","G","C"
a,t,g,c = "0,0,0,1","0,0,1,0","0,1,0,0","1,0,0,0"
ht = []

for line in fo:
    if v in line:
        continue
    if s in line:
        continue
   
    new_lines = line
    for j in range (len(new_lines)):
        if new_lines[j]==n_a:
            ht.append(a)

        if new_lines[j]==n_t:
            ht.append(t)

        if new_lines[j]==n_g:
            ht.append(g)

        if new_lines[j]==n_c:
            ht.append(c)
            
    sa = ","
    sa = sa.join(ht)
    print(sa+",0")
    ht.clear()
#-----------------------addition_with_target_class-----------
import pandas as pd
import numpy as np
import re
fo = open("healthy_di_tri_tetra_normalized.csv","r", encoding ="utf8", errors="ignore")
while True:
   line = fo.readline().rstrip()
   if not line:
      break
   line = line.rstrip()
   line = line.rstrip()
   print(line+",0")
#-----------------------------Replace_muiltiple_chracter------------------------------
import re
fo = open("healthy_166_filtered.tetra","r",encoding = "utf8", errors='ignore')
while True:
    line  = fo.readline()
    if not line:
        break
    a=re.sub('[:]|[-]','_',line).rstrip()
    print(a)
#--------------------------------------merge_multiplefile--------
import pandas as pd
df1 = pd.read_csv(r"pc.di")
df2 = pd.read_csv(r"pc.tri")
df3 = pd.read_csv(r"pc.tetra")
df = pd.merge(df1, df2,on = ['SeqID'])
df4 = pd.merge(df,df3, on = ['SeqID'])
df4.to_csv('CSV3.csv', index=None)
#-----------------------__delete_Coulumn_Normalization 
#----------Import library----------------------------------------
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
#----------------------------Read_-------------------------------
Data = pd.read_csv("test", sep=",")
headerList=(Data.columns[1:])
#------------------__delete__target-------------------------------
Data.drop(['SeqID'] ,inplace=True,index=None, axis=1)
#------------------Normalization-----------------
scaler = preprocessing.StandardScaler().fit(Data)
Data=scaler.transform(Data)
#---------------------------Addition_of_Target_Class-------------------------------------------------------------------------
df = pd.DataFrame(Data)
df['Class']=0
#--------------------write_output_result-------------------------------------------------------------------------------
pd.DataFrame(df).to_csv("test.csv", header=b,  index=False)  

