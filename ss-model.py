import argparse
import torch
import logging
import os

from src.model import SecStructPredictionHead
from src.dataset import create_dataloader
from src.utils import get_embed_dim

parser = argparse.ArgumentParser()

parser.add_argument("--device", default='cuda:0', type=str, help="Pytorch device to use (cpu or cuda)")
parser.add_argument("--embeddings_path", default='data/all_repr_ERNIE-RNA.h5', type=str, help="The path of the representations.")
parser.add_argument("--train_partition_path", default='./data/famfold-data/train-partition-0.csv', type=str, help="The path of the train partition.")
parser.add_argument("--val_partition_path", default='./data/famfold-data/valid-partition-0.csv', type=str, help="The path of the validation partition.")
parser.add_argument("--test_partition_path", default='./data/famfold-data/test-partition-0.csv', type=str, help="The path of the test partition.")
parser.add_argument("--batch_size", default=4, type=int, help="Batch size to use in forward pass.")
parser.add_argument("--max_epochs", default=10, type=int, help="Maximum number of training epochs.")
parser.add_argument("--patience", default=10, type=int, help="Epochs to wait before quiting training because of validation f1 not improving.")
parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate for the training.")
parser.add_argument("--out_path", default=10, type=str, help="Path to write predictions (base pairs of test partition), weights and logs")
parser.add_argument("--run_id", default="no-id", type=int, help="Run identifier")

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

train_loader = create_dataloader(
    args.embeddings_path,
    args.train_partition_path,
    args.batch_size,
    True
)

val_loader = create_dataloader(
    args.embeddings_path,
    args.val_partition_path,
    args.batch_size,
    False
)

test_loader = create_dataloader(
    args.embeddings_path,
    args.test_partition_path,
    args.batch_size,
    False
)

embed_dim = get_embed_dim(train_loader)
net = SecStructPredictionHead(embed_dim=embed_dim,device=args.device,lr=args.lr)
best_f1 = -1
patience_counter = 0
for epoch in range(args.max_epochs):
    logger.info(f"starting epoch {epoch}")
    train_metrics = net.fit(train_loader)
    val_metrics = net.test(val_loader)
    
    if epoch % 10 == 0:
        torch.save(net.state_dict(), os.path.join(args.out_path, f"weights{args.run_id}-epoch{epoch}.pmt"))
        logger.info(f"model saved at epoch {epoch}")
    if val_metrics["f1"] > best_f1:
        logger.info(f"f1 improved, was {best_f1} and now is {val_metrics['f1']}")
        patience_counter = 0
        best_f1 = val_metrics["f1"]
        torch.save(net.state_dict(), os.path.join(args.out_path, f"weights{args.run_id}-best.pmt"))
        logger.info(f"model saved at epoch {epoch}")
    else:
        logger.info(f"f1 has not improved, increasing patience counter")
        patience_counter+=1
        if patience_counter>args.patience:
            logger.info("exiting training loop, patience was reached")
            break
    msg = (
        f"epoch {epoch}:"
        + " ".join([f"train_{k} {v:.3f}" for k, v in train_metrics.items()])
        + " "
        + " ".join([f"val_{k} {v:.3f}" for k, v in val_metrics.items()])
    )
    logger.info(msg)

logger.info("loading model")
best_model = SecStructPredictionHead(embed_dim=embed_dim, device=args.device)
best_model.load_state_dict(torch.load(os.path.join(args.out_path, f"weights{args.run_id}-best.pmt"), map_location=torch.device(best_model.device)))
best_model.eval()
logger.info("running inference")
predictions = best_model.pred(test_loader)
predictions.to_csv(os.path.join(args.out_path, f"{args.run_id}.csv"), index=False)
logger.info(f"finished run {args.run_id}!")
