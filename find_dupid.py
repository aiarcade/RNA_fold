import pandas as pd


df = pd.read_csv('submission.csv')
df_no_duplicates = df.drop_duplicates(subset=['id'], keep='first')
df_no_duplicates.to_csv('submission_no_duplicates.csv', index=False)
# Identify duplicate values in the first column
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

# def length(min,max):
#     return max-min

# #df=pd.read_csv("../test_sequences.csv")
# df=pd.read_parquet("../train_data.parquet")
# reactivity_cols = df.filter(like='reactivity').columns
# df['reactivity'] = df[reactivity_cols].values.tolist()
# df = df.drop(columns=df.filter(like='reactivity_').columns,axis=1)
# df=df.reset_index(drop=True)
# df_exploded = df.explode('reactivity')
# print(df_exploded.columns)
# print(df_exploded['reactivity'].min())
# print(df_exploded['reactivity'].max())

#df['sequence_length']= df.apply(lambda row: length(row['id_min'], row['id_max']), axis=1)
# df['sequence_length']= df['sequence'].apply(len)
# length_counts = df['sequence_length'].value_counts()

# # Print the count of each unique length
# print(length_counts)

# df=pd.read_csv("../test_sequences.csv")
# #df=pd.read_parquet("../train_data.parquet")
# #df['sequence_length']= df.apply(lambda row: length(row['id_min'], row['id_max']), axis=1)
# df['sequence_length']= df['sequence'].apply(len)
# length_counts = df['sequence_length'].value_counts()

# # Print the count of each unique length
# print(length_counts)

