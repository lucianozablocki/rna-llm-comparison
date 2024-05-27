import torch
from torch.utils.data import Dataset
import pandas as pd
import h5py
import json

from utils import valid_mask

def bp2matrix(L, base_pairs):
    matrix = torch.zeros((L, L))

    for bp in base_pairs:
        # base pairs are 1-based
        matrix[bp[0] - 1, bp[1] - 1] = 1
        matrix[bp[1] - 1, bp[0] - 1] = 1

    return matrix

class EmbeddingDataset(Dataset):
    def __init__(
        self, dataset_path, embeddings_path): # min_len=0, max_len=512, verbose=False, cache_path=None, for_prediction=False, use_restrictions=False, **kargs):
        # self.max_len = max_len
        # self.verbose = verbose
        # if cache_path is not None and not os.path.isdir(cache_path):
        #     os.mkdir(cache_path)
        # self.cache = cache_path

        # Loading dataset
        data = pd.read_csv(dataset_path)
        # Loading representations
        self.embeddings = h5py.File(embeddings_path, 'r')
        # if for_prediction:
        #     assert (
        #         "sequence" in data.columns
        #         and "id" in data.columns
        #     ), "Dataset should contain 'id' and 'sequence' columns"

        # else:
        #     assert (
        #         ("base_pairs" in data.columns or "dotbracket" in data.columns)
        #         and "sequence" in data.columns
        #         and "id" in data.columns
        #     ), "Dataset should contain 'id', 'sequence' and 'base_pairs' or 'dotbracket' columns"

        #     if "base_pairs" not in data.columns and "dotbracket" in data.columns:
        #         data["base_pairs"] = data.dotbracket.apply(lambda x: str(dot2bp(x)))

        # data["len"] = data.sequence.str.len()

        # if max_len is None:
        #     max_len =
        # self.max_len = max(data.len)

        # datalen = len(data)

        # data = data[(data.len >= min_len) & (data.len <= max_len)]

        # if len(data) < datalen:
        #     print(
        #         f"From {datalen} sequences, filtering {min_len} < len < {max_len} we have {len(data)} sequences"
        #     )

        self.sequences = data.sequence.tolist()
        self.ids = data.id.tolist()
        print(f"keys are {len(self.embeddings)}")
        self.base_pairs = None
        self.base_pairs = [
            json.loads(data.base_pairs.iloc[i]) for i in range(len(data))
        ]
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        seq_id = self.ids[idx]
        # cache = f"{self.cache}/{seqid}.pk"
        # if (self.cache is not None) and os.path.isfile(cache):
        #     item = pickle.load(open(cache, "rb"))
        # else:
        sequence = self.sequences[idx]

        if seq_id not in self.embeddings:
            print(f"{seq_id} not present")
            raise
        # using [()] to read h5py Dataset as numpy array. related to how the file is created
        seq_emb = torch.from_numpy(self.embeddings[seq_id][()])
        L = seq_emb.shape[0] # embedding has size max_L x d
        # if L > self.max_len:
        #     print(f"{seq_id} too long: len {L} bigger than max L {self.max_len}")
        #     raise
        Mc = bp2matrix(L, self.base_pairs[idx])
        mask = valid_mask(sequence, L)
        return {"seq_id": seq_id, "seq_emb": seq_emb, "contact": Mc, "L": L, "sequence": sequence, "mask": mask} # seq_id, seq_emb, Mc, L

def pad_batch(batch):
    """batch is a list of dicts with keys: seqid, seq_emb, Mc, L, sequence, mask"""
    seq_ids, seq_embs, Mcs, Ls, sequences, masks = [[batch_elem[key] for batch_elem in batch] for key in batch[0].keys()]
    # should embedding_dim be computed for every batch?
    embedding_dim = seq_embs[0].shape[1] # seq_embs is a list of tensors of size L x d
    batch_size = len(batch)
    max_L = max(Ls)
    seq_embs_pad = torch.zeros(batch_size, max_L, embedding_dim)
    # cross entropy loss can ignore the -1s
    Mcs_pad = -torch.ones((batch_size, max_L, max_L), dtype=torch.long)
    masks_pad = torch.zeros((batch_size, max_L, max_L))
    for k in range(batch_size):
        seq_embs_pad[k, : Ls[k], :] = seq_embs[k]
        Mcs_pad[k, : Ls[k], : Ls[k]] = Mcs[k]
        masks_pad[k, : Ls[k], : Ls[k]] = masks[k]
    return {"seq_ids": seq_ids, "seq_embs_pad": seq_embs_pad, "contacts": Mcs_pad, "Ls": Ls, "sequences": sequences, "masks": masks_pad} # seq_ids, seq_embs_pad, Mcs, Ls


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
