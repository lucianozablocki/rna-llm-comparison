import pandas as pd 
import os 
import shutil 

# Params
emb_name = "one-hot"

df = pd.read_csv(f'data/ArchiveII.csv', index_col="id")
splits = pd.read_csv(f'data/ArchiveII_splits.csv', index_col="id")

device = "cuda:0"
emb_path = f"data/{emb_name}_ArchiveII.h5"

for k in range(5):
    train = df.loc[splits[(splits.fold==k) & (splits.partition!="test")].index]
    test = df.loc[splits[(splits.fold==k) & (splits.partition=="test")].index]
    data_path = f"data/archiveII_kfold/{k}/"
    out_path = f"results_{emb_name}_ArchiveII_kfold/{k}/"
    os.makedirs(data_path, exist_ok=True)
    shutil.rmtree(out_path, ignore_errors=True)
    os.makedirs(out_path, exist_ok=True)
    train.to_csv(f"{data_path}train.csv")
    test.to_csv(f"{data_path}test.csv")

    os.system(f"python train_model.py --device {device} --embeddings_path {emb_path} --train_partition_path {data_path}train.csv --out_path {out_path}")
    os.system(f"python test_model.py --device {device} --embeddings_path {emb_path} --test_partition_path {data_path}test.csv --out_path {out_path}")
