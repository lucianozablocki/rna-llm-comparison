import argparse
import pandas as pd
import extract_embedding # ERNIE-RNA
import h5py

parser = argparse.ArgumentParser()

parser.add_argument("--seqs_path", default='./data/ArchiveII.csv', type=str, help="The path of input RNA sequences.")
parser.add_argument("--device", default='cuda:0', type=str, help="Device to execute (either cpu or cuda).")
parser.add_argument("--output_path", default='./data', type=str, help="Folder to save embeddings file.")
parser.add_argument("--arg_overrides", default={ "data": './src/dict/' }, help="The path of vocabulary")
parser.add_argument("--ernie_rna_pretrained_checkpoint", default='./checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_pretrain.pt', type=str, help="The path of ERNIE-RNA checkpoint")

args = parser.parse_args()

data = pd.read_csv(args.seqs_path)

# we rely on the extract_embedding module of ERNIE-RNA to be available. we have to look for another way of invoking it
all_embeddings = extract_embedding.extract_embedding_of_ernierna(
   data["sequence"].tolist(),
   if_cls=False,
   arg_overrides=args.arg_overrides,
   pretrained_model_path=args.ernie_rna_pretrained_checkpoint,
   device=args.device
)

id_to_embedding = {}
# generate dictionary with seq ids as keys, and embedding tensors as values
for seq_id, embedding in zip(data['id'], all_embeddings):
  id_to_embedding[seq_id] = embedding
# an alternative way of converting it to a h5py file is to convert it to a pandas dataframe first
# pandas.from_dict() # orient='columns'

# h5, parquet, pickle, npy are possible file formats to store the representations
# we choose h5 here
with h5py.File(f'{args.output_path}/all_repr_ERNIE-RNA.h5', 'w') as hdf:
  for key, value in id_to_embedding.items():
    hdf.create_dataset(key, data=value)

# h5 file can be created as shown below if a pandas dataframe is chosen
# pandas_dataframe.to_hdf(DATA_PATH + "seqsim_f_all.h5", key='rnadist', mode='w')
# h5 file can be later read as this
# rnadist = pd.read_hdf(DATA_PATH + "rnadist_f_all.h5")
# instead of using [()] syntax as
# torch.from_numpy(self.embeddings[seq_id][()])
