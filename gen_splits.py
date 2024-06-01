import pandas as pd 
import os 

df = pd.read_csv('data/ArchiveII.csv')
df["fam"] = df["id"].str.split("_").str[0]
for fam in df["fam"].unique():
    os.makedirs(f"data/archiveII_famfold/{fam}/")
    df[df["fam"] == fam].to_csv(f'data/archiveII_famfold/{fam}/test.csv', index=False)
    train = df[df["fam"] != fam]
    train.to_csv(f'data/archiveII_famfold/{fam}/train.csv', index=False)