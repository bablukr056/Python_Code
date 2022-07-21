import re
f3 = open('PE_Segregated_070118.fasta','w')         #write new file as a PE_Segregated_070118.fasta

with open('nextprot_all.peff') as fasta:
  ok = False

  for line in fasta:
    line = line.rstrip()
    
    if ">" in line:
        if re.search("PE=5",line):
          id=line.rstrip()
          
          f3.write(id)
          f3.write("\n")
        
          ok = True
        else:
          ok = False
    else:
      if ok:
        f3.write(line)               #write fasta sequence which is below the accesin id 
        f3.write("\n")