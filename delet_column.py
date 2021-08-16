
# importing pandas module
import pandas as pd
  
# making data frame from csv file
data = pd.read_csv("morta_svm_combined.tsv", sep="\t" )
  
# dropping passed columns
data.pop("Name")
  
# display
print(data[0])
