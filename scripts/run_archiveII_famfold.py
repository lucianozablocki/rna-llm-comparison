import pandas as pd 
import os 
import shutil 

# Params
emb_name = "one-hot"

df = pd.read_csv(f'data/ArchiveII.csv')
device = "cuda:0"
emb_path = f"data/{emb_name}_ArchiveII.h5"
df["fam"] = df["id"].str.split("_").str[0]

for fam in df["fam"].unique():
    train = df[df["fam"] != fam]
    test = df[df["fam"] == fam]
    data_path = f"data/archiveII_famfold/{fam}/"
    out_path = f"results_{emb_name}_ArchiveII_famfold/{fam}/"
    os.makedirs(data_path, exist_ok=True)
    shutil.rmtree(out_path, ignore_errors=True)
    os.makedirs(out_path, exist_ok=True)
    train.to_csv(f"{data_path}train.csv", index=False)
    test.to_csv(f"{data_path}test.csv", index=False)

    os.system(f"python train_test_model.py --device {device} --embeddings_path {emb_path} --train_partition_path {data_path}train.csv --test_partition_path {data_path}test.csv --out_path {out_path}")
