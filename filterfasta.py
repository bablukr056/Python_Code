import re
fo = open("healthy_166.fasta","r",encoding = "utf8", errors='ignore')
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
