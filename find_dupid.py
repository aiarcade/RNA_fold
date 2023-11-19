import pandas as pd

# Read the CSV file into a DataFrame
# df = pd.read_csv('../submission.csv')

# # Identify duplicate values in the first column
# duplicate_rows = df[df.duplicated(subset=['id'], keep=False)]

# # Display the duplicate rows
# print("Duplicate Rows:")
# print(duplicate_rows)

# dataset=StructureProbTestDataset("../testdata/")
# test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
# mins=[]
# maxs=[]

# for x,ids,fid in test_dataloader:
#     id_min=int(ids[0][0])
#     id_max=int(ids[0][1])
#     if id_min not in mins:
#         mins.append(id_min)
#     else:
#         print("found duplicate",id_min,id_max,fid)
#     if id_max not in maxs:
#         maxs.append(id_max) 
#     else:
#         print("found duplicate",id_min,id_max,fid)


# df = pd.read_csv('../corrected_submission.csv')

# id=0
# for index, row in df.iterrows():
#     if int(row['id'])!=id:
#         print("Mismatch",id,row)
#         break
#     else:
#         id=id+1

def length(min,max):
    return max-min

df=pd.read_csv("../test_sequences.csv")
df['sequence_length']= df.apply(lambda row: length(row['id_min'], row['id_max']), axis=1)
length_counts = df['sequence_length'].value_counts()

# Print the count of each unique length
print(length_counts)


