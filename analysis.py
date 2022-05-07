import re
import pandas as pd
import numpy as np
#---------------------

fo = open("1","r")
i=0
while True:
    line  = fo.readline()
    if not line:
        break
    line=re.split(';',line)
    if int(line[2])=='1':
        i=i.count(1)
        print(i)
