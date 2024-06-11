import pandas as pd 
import os 
import shutil 

# Params
emb_name = "RNAErnie"

df = pd.read_csv(f'data/bpRNA.csv', index_col="id")
splits = pd.read_csv(f"data/bpRNA_splits.csv", index_col="id")
device = "cuda"
emb_path = f"../insync/lncRNA/LLM-RNA/embeddings/{emb_name}_bpRNA.h5"
print(emb_path)

train = pd.concat((df.loc[splits.partition=="TR0"], df.loc[splits.partition=="VL0"])) 
test = df.loc[splits.partition=="TS0"]
data_path = f"data/archiveII_bprna/"
out_path = f"results_{emb_name}_bprna/"
os.makedirs(data_path, exist_ok=True)
shutil.rmtree(out_path, ignore_errors=True)
os.makedirs(out_path, exist_ok=True)
train.to_csv(f"{data_path}train.csv")
test.to_csv(f"{data_path}test.csv")

os.system(f"python train_test_model.py --device {device} --embeddings_path {emb_path} --train_partition_path {data_path}train.csv --test_partition_path {data_path}test.csv --out_path {out_path}")
