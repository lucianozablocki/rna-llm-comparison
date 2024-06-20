import pandas as pd 
import os 
import shutil 

# Params
emb_name = "one-hot"

df = pd.read_csv(f'data/pdb.csv', index_col="id")
splits = pd.read_csv(f"data/pdb_splits.csv", index_col="id")
device = "cuda"
emb_path = f"embeddings/{emb_name}_pdb.h5"
print(emb_path)

train = df.loc[splits.partition=="train"] 
test = df.loc[splits.partition=="test"]
data_path = f"data/pdb_splits/"
out_path = f"results_{emb_name}_pdb/"
os.makedirs(data_path, exist_ok=True)
shutil.rmtree(out_path, ignore_errors=True)
os.makedirs(out_path, exist_ok=True)
train.to_csv(f"{data_path}train.csv")
test.to_csv(f"{data_path}test.csv")

os.system(f"python train_test_model.py --device {device} --embeddings_path {emb_path} --train_partition_path {data_path}train.csv --test_partition_path {data_path}test.csv --out_path {out_path}")
