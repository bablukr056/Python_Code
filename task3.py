import re

with open('PE_Segregated_070118.fasta') as fasta:
  for line in fasta:
      line = line.rstrip()
      if '>' in line:
        id=line
        line=re.split(":| ",line) #split based on : and space
        acc=line[1]                 #hadder
      
        a=(next(fasta)).rstrip()
        a=a.split("\n")[0]
        b = re.split('(R|K)', a)
        
        c = []

        for i in range(len(b)-1):
          if len(b[i]) >= 6 and len(b[i]) <= 34 and (b[i+1] == 'R' or b[i+1] == 'K'):
            c.append(b[i] + b[i+1])
        
        d = ';'.join(c)
        
        print(f'{acc}\t{len(c)}\t{d}')