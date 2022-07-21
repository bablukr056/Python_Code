import re

data1_output_dic={}

with open ("data1_output.txt","r") as file1, open("data2.txt","r") as file2:


    for line_1 in file1:
        a=line_1.rstrip().rsplit("\t")
        x=a[0]
        y="\t"+a[0]+"\t"+a[1]+"\t"+a[2]
        data1_output_dic[x]=y

    for line_2 in file2:

        b=line_2.rstrip().rsplit('\t')
        b1=str(b[0])
        b2 = b1.rsplit('|')
        #print(b2[2])
        if b2[2] in data1_output_dic:

            print(b[0]+"\t"+b[1]+"\t"+b[2]+"\t"+b[3]+data1_output_dic[b2[2]])
        else:
            print(b[0]+"\t"+b[1]+"\t"+b[2]+"\t"+b[3]+"\t"+"-"+"\t"+"-"+"\t"+"-")