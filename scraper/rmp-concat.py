# importing the required modules
import glob
import pandas as pd
  
# specifying the path to csv files
path = "/Users/peterlee/Downloads/rmp data"
  
# csv files in the path
files = glob.glob(path + "/*.csv")
  
# defining an empty list to store 
# content
data_frame = pd.DataFrame()
content = []
  

# df = pd.read_csv(files[0], index_col=None)
# print(df)

# checking all the csv files in the 
# specified path
counter = 0
for filename in files:
    
    # reading content of csv file
    # content.append(filename)
    print(str(counter) + "   ->" + filename)
    df = pd.read_csv(filename, index_col=None)
    content.append(df)
    counter += 1
    
  
# converting content to data frame
data_frame = pd.concat(content)
print(data_frame)

data_frame.to_csv('rmp-concat-data.csv')