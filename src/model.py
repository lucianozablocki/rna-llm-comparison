import torch.nn as nn
import torch
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import LinearLR
import pandas as pd
from metrics import contact_f1
from utils import mat2bp, postprocessing, outer_concat


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
        self.threshold = 0.5
        self.linear_in = nn.Linear(embed_dim * 2, conv_dim)
        self.resnet = ResNet2D(conv_dim, num_blocks, kernel_size)
        self.conv_out = nn.Conv2d(conv_dim, 1, kernel_size=kernel_size, padding="same")
        self.device = device
        self.class_weight = torch.tensor([negative_weight, 1.0]).float().to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.lr_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=15)

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
        x = outer_concat(x, x) # B x L x F => B x L x L x 2F

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
        self.lr_scheduler.step()
        loss_acum /= len(loader)
        f1_acum /= len(loader)
        return {"loss": loss_acum, "f1": f1_acum}

    def test(self, loader):
        loss_acum = 0
        f1_acum = 0
        for batch in loader:
            X = batch["seq_embs_pad"].to(self.device)
            y = batch["contacts"].to(self.device)
            y_pred = self(X)
            loss = self.loss_func(y_pred, y)
            loss_acum += loss.item()
            f1_acum += contact_f1(y.cpu(), y_pred.detach().cpu(), batch["Ls"], method="triangular")
        loss_acum /= len(loader)
        f1_acum /= len(loader)
        return {"loss": loss_acum, "f1": f1_acum}

    def pred(self, loader):
        # self.eval()

        # if self.verbose:
        #     loader = tqdm(loader)

        predictions = [] 
        # with tr.no_grad():
        for batch in loader: 
            
            Ls = batch["Ls"]
            seq_ids = batch["seq_ids"]
            sequences = batch["sequences"]
            X = batch["seq_embs_pad"].to(self.device)

            y_pred = self(X)
            
            # if isinstance(y_pred, tuple):
            #     y_pred = y_pred[0]

            y_pred_post = postprocessing(y_pred.cpu(), batch["masks"])
            for k in range(len(y_pred_post)):
                # if logits:
                #     logits_list.append(
                #         (seqid[k],
                #             y_pred[k, : lengths[k], : lengths[k]].squeeze().cpu(),
                #             y_pred_post[k, : lengths[k], : lengths[k]].squeeze()
                #         ))
                predictions.append((
                    seq_ids[k],
                    sequences[k],
                    mat2bp(
                        y_pred_post[k, : Ls[k], : Ls[k]].squeeze()
                    )                         
                ))
        predictions = pd.DataFrame(predictions, columns=["id", "sequence", "base_pairs"])

        return predictions
