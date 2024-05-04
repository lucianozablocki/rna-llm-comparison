import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch
from sklearn.metrics import f1_score
from torch.nn.functional import cross_entropy
from torch.utils.data import Dataset
import pandas as pd
import h5py
import json

def contact_f1(ref_batch, pred_batch, Ls, th=0.5, reduce=True, method="triangular"):
    """Compute F1 from base pairs. Input goes to sigmoid and then thresholded"""
    f1_list = []

    if type(ref_batch) == float or len(ref_batch.shape) < 3:
        ref_batch = [ref_batch]
        pred_batch = [pred_batch]
        L = [L]

    for ref, pred, l in zip(ref_batch, pred_batch, Ls):
        # ignore padding
        ind = torch.where(ref != -1)
        pred = pred[ind].view(l, l)
        ref = ref[ind].view(l, l)

        # pred goes from -inf to inf
        pred = torch.sigmoid(pred)
        pred[pred<=th] = 0

        if method == "triangular":
            f1 = f1_triangular(ref, pred>th)
        elif method == "shift":
            raise NotImplementedError
        else:
            raise NotImplementedError

        f1_list.append(f1)

    if reduce:
        return torch.tensor(f1_list).mean().item()
    else:
        return torch.tensor(f1_list)


def f1_triangular(ref, pred):
    """Compute F1 from the upper triangular connection matrix"""
    # get upper triangular matrix without diagonal
    ind = torch.triu_indices(ref.shape[0], ref.shape[1], offset=1)

    ref = ref[ind[0], ind[1]].numpy().ravel()
    pred = pred[ind[0], ind[1]].numpy().ravel()

    return f1_score(ref, pred, zero_division=0)

def _outer_concat(t1: torch.Tensor, t2: torch.Tensor):
    # t1, t2: shape = B x L x E
    assert t1.shape == t2.shape, f"Shapes of input tensors must match! ({t1.shape} != {t2.shape})"

    seq_len = t1.shape[1]
    a = t1.unsqueeze(-2).expand(-1, -1, seq_len, -1)
    b = t2.unsqueeze(-3).expand(-1, seq_len, -1, -1)

    return torch.concat((a, b), dim=-1)

class ResNet2DBlock(nn.Module):
    def __init__(self, embed_dim, kernel_size=3, bias=False):
        super().__init__()

        # Bottleneck architecture
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, bias=bias),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=kernel_size, bias=bias, padding="same"),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1, bias=bias),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        residual = x

        x = self.conv_net(x)
        x = x + residual

        return x

class ResNet2D(nn.Module):
    def __init__(self, embed_dim, num_blocks, kernel_size=3, bias=False):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ResNet2DBlock(embed_dim, kernel_size, bias=bias) for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x

class SecStructPredictionHead(nn.Module):
    def __init__(self, embed_dim, num_blocks=2, conv_dim=64, kernel_size=3, negative_weight=0.1, device='cpu', lr=1e-5):
        super().__init__()
        self.lr = lr
        # self.loss = nn.BCEWithLogitsLoss()
        self.threshold = 0.5
        self.linear_in = nn.Linear(embed_dim * 2, conv_dim)
        self.resnet = ResNet2D(conv_dim, num_blocks, kernel_size)
        self.conv_out = nn.Conv2d(conv_dim, 1, kernel_size=kernel_size, padding="same")
        self.device = device
        self.class_weight = torch.tensor([negative_weight, 1.0]).float().to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.to(device)

    def loss_func(self, yhat, y):
        """yhat and y are [N, M, M]"""
        y = y.view(y.shape[0], -1)
        # yhat, y0 = yhat  # yhat is the final ouput and y0 is the cnn output

        yhat = yhat.view(yhat.shape[0], -1)
        # y0 = y0.view(y0.shape[0], -1)

        # # Add l1 loss, ignoring the padding
        # l1_loss = tr.mean(tr.relu(yhat[y != -1]))

        # yhat has to be shape [N, 2, L].
        yhat = yhat.unsqueeze(1)
        # yhat will have high positive values for base paired and high negative values for unpaired
        yhat = torch.cat((-yhat, yhat), dim=1)

        # y0 = y0.unsqueeze(1)
        # y0 = tr.cat((-y0, y0), dim=1)
        # error_loss1 = cross_entropy(y0, y, ignore_index=-1, weight=self.class_weight)

        error_loss = cross_entropy(yhat, y, ignore_index=-1, weight=self.class_weight)


        loss = (
            error_loss
            # + self.loss_beta * error_loss1
            # + self.loss_l1 * l1_loss
        )
        return loss

    def forward(self, x):
        # print(f"before ellipsis: {x.shape}") batch_size x L x d
        # x = x[..., 1:-1, :] # this line was commented out, we want to use embeddings as they come from the LLM
        # print(f"after ellipsis: {x.shape}") batch_size x L-2 x d
        x = _outer_concat(x, x) # B x L x F => B x L x L x 2F

        x = self.linear_in(x)
        x = x.permute(0, 3, 1, 2) # B x L x L x E  => B x E x L x L

        x = self.resnet(x)
        x = self.conv_out(x)
        x = x.squeeze(-3) # B x 1 x L x L => B x L x L

        # Symmetrize the output
        x = torch.triu(x, diagonal=1)
        x = x + x.transpose(-1, -2)

        return x.squeeze(-1)

    def fit(self, loader):
      loss_acum = 0
      f1_acum = 0
      for batch in loader:
        X = batch["seq_embs_pad"].to(self.device)
        y = batch["contacts"].to(self.device)
        y_pred = self(X)
        # print(f"y_pred size: {y_pred.shape}") # torch.Size([4, 512, 512])
        # print(f"y size: {y.shape}") # torch.Size([4, 512, 512])
        loss = self.loss_func(y_pred, y)
        loss_acum += loss.item()
        f1_acum += contact_f1(y.cpu(), y_pred.detach().cpu(), batch["Ls"], method="triangular")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
      loss_acum /= len(loader)
      f1_acum /= len(loader)
      return {"loss": loss_acum, "f1": f1_acum}

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

        return {"seq_id": seq_id, "seq_emb": seq_emb, "contact": Mc, "L": L} # seq_id, seq_emb, Mc, L

def pad_batch(batch):
    """batch is a list of (seqid, seq_emb, Mc, L)"""
    seq_ids, seq_embs, Mcs, Ls = [[batch_elem[key] for batch_elem in batch] for key in batch[0].keys()]
    embedding_dim = seq_embs[0].shape[1] # seq_embs is a list of tensors of size L x d
    batch_size = len(batch)
    max_L = max(Ls)
    seq_embs_pad = torch.zeros(batch_size, max_L, embedding_dim)
    # cross entropy loss can ignore the -1s
    Mcs_pad = -torch.ones((batch_size, max_L, max_L), dtype=torch.long)
    for k in range(batch_size):
        seq_embs_pad[k, : Ls[k], :] = seq_embs[k]
        Mcs_pad[k, : Ls[k], : Ls[k]] = Mcs[k]
    return {"seq_ids": seq_ids, "seq_embs_pad": seq_embs_pad, "contacts": Mcs_pad, "Ls": Ls} # seq_ids, seq_embs_pad, Mcs, Ls

parser = argparse.ArgumentParser()

parser.add_argument("--embeddings_path", default='../ERNIE-RNA/data/all_repr_ERNIE-RNA.h5', type=str, help="The path of the representations.")
parser.add_argument("--train_partition_path", default='./data/famfold-data/train-partition-0.csv', type=str, help="The path of the train partition.")
parser.add_argument("--batch_size", default=4, type=int, help="Batch size to use in forward pass.")
parser.add_argument("--shuffle", default=False, type=bool, help="Wheter to shuffle the data upon Dataloder creation.")
parser.add_argument("--max_epochs", default=10, type=int, help="Maximum number of training epochs.")

args = parser.parse_args()

train_dataset = EmbeddingDataset(
  embeddings_path=args.embeddings_path,
  dataset_path=args.train_partition_path,
)

# ToDo
# val_dataset = EmbeddingDataset(
#   embeddings_path=f"{my_drive_path}/all_repr_ERNIE-RNA.h5",
#   partition_path=f"{root_path}/{val_partition_path}",
# )

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=args.shuffle,
    collate_fn=pad_batch,
)

# ToDo
# val_loader = torch.utils.data.DataLoader(
#     val_dataset,
#     batch_size=batch_size,
#     shuffle=shuffle,
#     collate_fn=pad_batch,
# )

# grab an element from the loader (a batch)
batch_elem = next(iter(train_loader))
# query for `seq_embs_pad` key (containing the embedding representations of all the sequences in the batch)
# whose size will be batch_size x L x d
embed_dim = batch_elem["seq_embs_pad"].shape[2]
net = SecStructPredictionHead(embed_dim=embed_dim,device='cuda:0')
for epoch in range(args.max_epochs):
  train_metrics = net.fit(train_loader)
  # val_metrics = net.test(val_loader)
  msg = (
    f"epoch {epoch}:"
    + " ".join([f"train_{k} {v:.3f}" for k, v in train_metrics.items()])
    # + " "
    # + " ".join([f"val_{k} {v:.3f}" for k, v in val_metrics.items()])
  )
  print(msg)
