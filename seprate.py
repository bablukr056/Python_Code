import pandas as pd
import numpy as np
from itertools import islice
import re
#________________________________________________________
fo = open("Gene_Count.csv","r")

while True:
    line  = fo.readline()
    if not line:
        break
    line=re.split(',',line)
    a=line[0]+"\t"+line[1]+"\t"+line[2]+"\t"+line[3]+"\t"+line[4]+"\t"+line[5]+"\t"+line[6]+"\t"+line[7]+"\t"+line[8]+"\t"+line[9]+"\t"+line[10]+"\t"+line[11]+"\t"+line[12]+"\t"+line[13]+line[14]+"\t"+line[15]+"\t"+line[16]+"\t"+line[17]+"\t"+line[18]+"\t"+line[19]+"\t"+line[20]+"\t"+line[21]+"\t"+line[22]

    print(a.rstrip())
    #if re.search('1',line[11]):
        #a=line[0]+"\t"+line[1]+"\t"+line[2]+"\t"+line[3]+"\t"+line[4]+"\t"+line[5]+"\t"+line[6]+"\t"+line[7]+"\t"+line[8]+"\t"+line[9]+"\t"+line[10]+"\t"+line[11]+"\t"+line[12]+"\t"+line[13]
        #a=a.rstrip()
        #print(a)
