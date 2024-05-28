import torch
from torch.utils.data import Dataset
import pandas as pd
import h5py
import json

def bp2matrix(L, base_pairs):
    matrix = torch.zeros((L, L))
    # base pairs are 1-based
    bp = torch.tensor(base_pairs) - 1
    if len(bp.shape) == 2:
        matrix[bp[:, 0], bp[:, 1]] = 1
        matrix[bp[:, 1], bp[:, 0]] = 1

    return matrix

class EmbeddingDataset(Dataset):
    def __init__(
        self, dataset_path, embeddings_path): 

        # Loading dataset
        data = pd.read_csv(dataset_path)
        self.sequences = data.sequence.tolist()
        self.ids = data.id.tolist()
        
        # Loading representations
        self.embeddings = {}
        embeddings = h5py.File(embeddings_path, 'r')
        # keep only sequeneces in dataset_path
        for seq_id in self.ids:
            self.embeddings[seq_id] = torch.from_numpy(embeddings[seq_id][()])
        
        print(f"keys are {len(self.embeddings)}")
        self.base_pairs = None
        self.base_pairs = [
            json.loads(data.base_pairs.iloc[i]) for i in range(len(data))
        ]
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        seq_id = self.ids[idx]
        sequence = self.sequences[idx]

        if seq_id not in self.embeddings:
            print(f"{seq_id} not present")
            raise
        seq_emb = self.embeddings[seq_id]
        L = len(sequence) 

        Mc = bp2matrix(L, self.base_pairs[idx])
        return {"seq_id": seq_id, "seq_emb": seq_emb, "contact": Mc, "L": L, "sequence": sequence} 

def pad_batch(batch):
    """batch is a list of dicts with keys: seqid, seq_emb, Mc, L, sequence, mask"""
    seq_ids, seq_embs, Mcs, Ls, sequences = [[batch_elem[key] for batch_elem in batch] for key in batch[0].keys()]
    # should embedding_dim be computed for every batch?
    embedding_dim = seq_embs[0].shape[1] # seq_embs is a list of tensors of size L x d
    batch_size = len(batch)
    max_L = max(Ls)
    seq_embs_pad = torch.zeros(batch_size, max_L, embedding_dim)
    # cross entropy loss can ignore the -1s
    Mcs_pad = -torch.ones((batch_size, max_L, max_L), dtype=torch.long)
    for k in range(batch_size):
        seq_embs_pad[k, : Ls[k], :] = seq_embs[k][:Ls[k], :]
        Mcs_pad[k, : Ls[k], : Ls[k]] = Mcs[k]
    return {"seq_ids": seq_ids, "seq_embs_pad": seq_embs_pad, "contacts": Mcs_pad, "Ls": Ls, "sequences": sequences}


def create_dataloader(embeddings_path, partition_path, batch_size, shuffle, collate_fn=pad_batch):
    dataset = EmbeddingDataset(
        embeddings_path=embeddings_path,
        dataset_path=partition_path,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
