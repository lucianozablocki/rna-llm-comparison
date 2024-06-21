import numpy as np
import argparse
import torch
import pandas as pd
import logging
import os

from src.model import SecStructPredictionHead
from src.dataset import create_dataloader
from src.utils import get_embed_dim


parser = argparse.ArgumentParser()

parser.add_argument("--device", default='cuda:0', type=str, help="Pytorch device to use (cpu or cuda)")
parser.add_argument("--embeddings_path", default='data/all_repr_ERNIE-RNA.h5', type=str, help="The path of the representations.")
parser.add_argument("--test_partition_path", default='data/archiveII_famfold/grp1/test.csv', type=str, help="The path of the test partition.")
parser.add_argument("--batch_size", default=2, type=int, help="Batch size to use in forward pass.")
parser.add_argument("--out_path", default="./results", type=str, help="Path to read model from, and to write predictions/metrics/logs")

args = parser.parse_args()

logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s.%(lineno)d - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler(os.path.join(args.out_path, f'log-test.txt'), mode='w'),
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"Run on {args.out_path}, with device {args.device} and embeddings {args.embeddings_path}")
logger.info(f"Testing with file: {args.test_partition_path}")

test_loader = create_dataloader(
    args.embeddings_path,
    args.test_partition_path,
    args.batch_size,
    False
)
embed_dim = get_embed_dim(test_loader)
best_model = SecStructPredictionHead(embed_dim=embed_dim, device=args.device)
best_model.load_state_dict(torch.load(os.path.join(args.out_path, f"weights.pmt"), map_location=torch.device(best_model.device)))
best_model.eval()

logger.info("running inference")
metrics = best_model.test(test_loader)
metrics = {f"test_{k}": v for k, v in metrics.items()}
logger.info(" ".join([f"{k}: {v:.3f}" for k, v in metrics.items()]))

pd.set_option('display.float_format','{:.3f}'.format)
pd.DataFrame([metrics]).to_csv(os.path.join(args.out_path, f"metrics_test.csv"), index=False)

logger.info("predicting")
predictions = best_model.pred(test_loader)
predictions.to_csv(os.path.join(args.out_path, f"preds.csv"), index=False)