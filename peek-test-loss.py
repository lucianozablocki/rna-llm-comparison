import argparse
import torch
import pandas as pd
import logging
import os
import ast

from src.model import SecStructPredictionHead
from src.dataset import create_dataloader
from src.utils import get_embed_dim
from src.metrics import f1_shift


parser = argparse.ArgumentParser()

parser.add_argument("--device", default='cuda:0', type=str, help="Pytorch device to use (cpu or cuda)")
parser.add_argument("--embeddings_path", default='data/all_repr_ERNIE-RNA.h5', type=str, help="The path of the representations.")
parser.add_argument("--test_partition_path", default='./data/famfold-data/test-partition-0.csv', type=str, help="The path of the test partition.")
parser.add_argument("--batch_size", default=2, type=int, help="Batch size to use in forward pass.")
parser.add_argument("--out_path", default="./results", type=str, help="Path to write predictions (base pairs of test partition), weights and logs")
parser.add_argument("--run_id", default="no-id", type=int, help="Run identifier")
parser.add_argument("--checkpoints", type=int, help="Amount of saved checkpoint")

args = parser.parse_args()

logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s.%(lineno)d - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler(os.path.join(args.out_path, f'log-{args.run_id}.txt'), mode='w'),
    ]
)
logger = logging.getLogger(__name__)

test_loader = create_dataloader(
    args.embeddings_path,
    args.test_partition_path,
    args.batch_size,
    False
)
embed_dim = get_embed_dim(test_loader)
data = pd.read_csv(args.test_partition_path, index_col="id")

for i in range(args.checkpoints):
    f1_acum=0
    logger.info(f"loading model for epoch {i*10}")
    best_model = SecStructPredictionHead(embed_dim=embed_dim, device=args.device)
    best_model.load_state_dict(torch.load(os.path.join(args.out_path, f"weights{args.run_id}-epoch{i*10}.pmt"), map_location=torch.device(best_model.device)))
    best_model.eval()
    logger.info("running inference")
    predictions = best_model.pred(test_loader)
    predictions = predictions.set_index("id")
    for seq_id in predictions.index:
        prediction = predictions.loc[seq_id]["base_pairs"]
        ref = ast.literal_eval(data.loc[seq_id]["base_pairs"])
        _, _, f1 = f1_shift(ref, prediction)
        f1_acum+=f1
    f1_acum/=len(predictions)
    logger.info(f"f1 for epoch {i*10} is {f1_acum}")


f1_acum=0
logger.info(f"loading model for best epoch")
best_model = SecStructPredictionHead(embed_dim=embed_dim, device=args.device)
best_model.load_state_dict(torch.load(os.path.join(args.out_path, f"weights{args.run_id}-best.pmt"), map_location=torch.device(best_model.device)))
best_model.eval()
logger.info("running inference")
predictions = best_model.pred(test_loader)
predictions = predictions.set_index("id")
for seq_id in predictions.index:
    prediction = predictions.loc[seq_id]["base_pairs"]
    ref = ast.literal_eval(data.loc[seq_id]["base_pairs"])
    _, _, f1 = f1_shift(ref, prediction)
    f1_acum+=f1
f1_acum/=len(predictions)
logger.info(f"f1 for best epoch is {f1_acum}")
