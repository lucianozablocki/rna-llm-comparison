import argparse
import torch
import logging
import os

from src.model import SecStructPredictionHead
from src.dataset import create_dataloader
from src.utils import get_embed_dim

parser = argparse.ArgumentParser()

parser.add_argument("--device", default='cuda:0', type=str, help="Pytorch device to use (cpu or cuda)")
parser.add_argument("--embeddings_path", default='data/all_repr_archiveii_RiNALMo.h5', type=str, help="The path of the representations.")
parser.add_argument("--train_partition_path", default='data/archiveII_famfold/grp1/train.csv', type=str, help="The path of the train partition.")
parser.add_argument("--test_partition_path", default='data/archiveII_famfold/grp1/test.csv', type=str, help="The path of the test partition.")
parser.add_argument("--batch_size", default=4, type=int, help="Batch size to use in forward pass.")
parser.add_argument("--max_epochs", default=15, type=int, help="Maximum number of training epochs.")
parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate for the training.")
parser.add_argument("--out_path", type=str, help="Path to write predictions (base pairs of test partition), weights and logs")

args = parser.parse_args()

os.makedirs(args.out_path, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s.%(lineno)d - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler(os.path.join(args.out_path, f'log.txt'), mode='w'),
    ]
)
logger = logging.getLogger(__name__)

train_loader = create_dataloader(
    args.embeddings_path,
    args.train_partition_path,
    args.batch_size,
    True
)

test_loader = create_dataloader(
    args.embeddings_path,
    args.test_partition_path,
    args.batch_size,
    False
)

embed_dim = get_embed_dim(train_loader)
net = SecStructPredictionHead(embed_dim=embed_dim, device=args.device, lr=args.lr)

for epoch in range(args.max_epochs):
    logger.info(f"starting epoch {epoch}")
    train_metrics = net.fit(train_loader)
    
    msg = (
        f"epoch {epoch}:"
        + " ".join([f"train_{k} {v:.3f}" for k, v in train_metrics.items()])
    )
    logger.info(msg)

    # TODO esto va fuera del loop de epochs
    logger.info("running inference")
    test_metrics = net.test(test_loader)
    for k in test_metrics:
        logger.info(f"test_{k} {test_metrics[k]:.3f}")

predictions = net.pred(test_loader)
predictions.to_csv(os.path.join(args.out_path, f"preds.csv"), index=False)
logger.info(f"finished run!")
