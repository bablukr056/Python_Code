for line in open('gencode.v38.lncRNA_transcripts.fa'):
    if '>' in line:
        print (line)
