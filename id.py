import random
n = random.randint(1,251)

def files():
    n = 0
    while True:
        n += 1
        yield open('divya%d.sdf' % n, 'w')

pat = 'RDKit'
fs = files()
outfile = next(fs)

with open("ChEMBL_Antibacterial_Library.sdf") as infile:
    for line in infile:
        if pat not in line:
            outfile.write(line)
        else:
            items = line.split(pat)
            outfile.write(items[0])
            for item in items[1:]:
                outfile = next(fs)
                outfile.write(str(n)+"\n"+pat + item)
