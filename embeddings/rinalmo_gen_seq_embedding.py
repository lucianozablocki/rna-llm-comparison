import torch
from rinalmo.pretrained import get_pretrained_model # RiNALMo
import argparse
import pandas as pd
import math
import h5py

parser = argparse.ArgumentParser()

parser.add_argument("--seqs_path", default='./data/ArchiveII.csv', type=str, help="The path of input RNA sequences.")
parser.add_argument("--device", default='cuda:0', type=str, help="Device to execute (either cpu or cuda).")
parser.add_argument("--output_path", default='./data', type=str, help="Folder to save embeddings file.")
parser.add_argument("--batch_size", default='100', type=int, help="Batch size to use")

args = parser.parse_args()

print("reading csv")
data = pd.read_csv(args.seqs_path)

print("loading model")
model, alphabet = get_pretrained_model(model_name="giga-v1")
model = model.to(device=args.device)
model.eval()

print("tokenizing")
tokens = torch.tensor(alphabet.batch_tokenize(data["sequence"].tolist()), dtype=torch.int64, device=args.device)

print("generating repr")
# as passing computing the representations for all the dataset at once
# produced a CUDAOutOfMemoryError, we make it by batches
total_seqs = len(data)
total_iterations = math.floor(total_seqs/args.batch_size)
rem=total_seqs-(total_iterations*args.batch_size)

all_embeddings = []
for i in range(total_iterations):
  lower_limit = i*args.batch_size
  with torch.no_grad(), torch.cuda.amp.autocast():
    outputs = model(tokens[lower_limit:args.batch_size+lower_limit])
  all_embeddings.extend(outputs["representation"])
if rem:
  last_idx=(args.batch_size+(args.batch_size*(total_iterations-1)))
  with torch.no_grad(), torch.cuda.amp.autocast():
    outputs = model(tokens[last_idx:last_idx+rem])
  all_embeddings.extend(outputs["representation"])

print("generating output dict")
id_to_embedding = {}
# generate dictionary with seq ids as keys, and embedding tensors as values
for seq_id, embedding in zip(data['id'], all_embeddings):
  id_to_embedding[seq_id] = embedding
# an alternative way of converting it to a h5py file is to convert it to a pandas dataframe first
# pandas.from_dict() # orient='columns'

print(f"total number of sequences: {len(id_to_embedding)}")

# h5, parquet, pickle, npy are possible file formats to store the representations
# we choose h5 here
with h5py.File(f'{args.output_path}/all_repr_archiveii_RiNALMo.h5', 'w') as hdf:
  for key, value in id_to_embedding.items():
    hdf.create_dataset(key, data=value.cpu())