import os

import numpy as np
from tqdm import tqdm
from pathlib import Path
from loguru import logger

import torch
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score


from .model import Net
from .dataloader import YooChooseDataset

if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"


class YooChooseGNN:
    def __init__(
        self, dataset_dir, model_save_path, epochs=1, lr=0.005, batch_size=512,
    ):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.dataset = YooChooseDataset(root=dataset_dir)
        self._dataset_utils()
        self.device = torch.device(DEVICE)
        self.model = Net(self.dataset.num_embeddings, embed_dim=128).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.crit = torch.nn.BCELoss()
        self.model_save_path = model_save_path

    def _dataset_utils(self):
        len_ds = len(self.dataset)
        logger.info(f"Shuffling dataset with {len_ds} samples.")
        self.dataset = self.dataset.shuffle()
        self.train_dataset = self.dataset[: int(0.8 * len_ds)]
        self.val_dataset = self.dataset[int(0.8 * len_ds) : int(0.9 * len_ds)]
        self.test_dataset = self.dataset[int(0.9 * len_ds) :]
        logger.info(
            "Training samples: {}, Validation samples: {}, Testing samples: {}".format(
                len(self.train_dataset), len(self.val_dataset), len(self.test_dataset),
            )
        )

    def train(self):
        self.model.train()
        epoch_loss_all = 0
        for data in tqdm(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            label = data.y.to(self.device)
            loss = self.crit(output, label)
            loss.backward()
            epoch_loss_all += data.num_graphs * loss.item()
            self.optimizer.step()
        return epoch_loss_all / len(self.train_dataset)

    def evaluate(self, loader):
        self.model.eval()
        predictions = []
        labels = []

        with torch.no_grad():
            for data in loader:

                data = data.to(self.device)
                pred = self.model(data).detach().cpu().numpy()

                label = data.y.detach().cpu().numpy()
                predictions.append(pred)
                labels.append(label)

        predictions = np.hstack(predictions)
        labels = np.hstack(labels)

        return roc_auc_score(labels, predictions)

    def save_model(self):
        logger.info(f"Saving Model in path {self.model_save_path}...")
        torch.save(self.model.state_dict(), self.model_save_path)

    def pipeline(self):
        self.train_loader, self.val_loader, self.test_loader = (
            DataLoader(x, batch_size=self.batch_size)
            for x in [self.train_dataset, self.val_dataset, self.test_dataset]
        )
        logger.info(f"Training for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch+1} started.")
            loss = self.train()
            logger.info("Generating Metrics...")
            train_acc = self.evaluate(self.train_loader)
            val_acc = self.evaluate(self.val_loader)
            test_acc = self.evaluate(self.test_loader)
            logger.info(
                "Epoch: {}, Loss: {:.5f}, Train Auc: {:.5f}, Val Auc: {:.5f}, Test Auc: {:.5f}".format(
                    epoch, loss, train_acc, val_acc, test_acc
                )
            )
        self.save_model()


if __name__ == "__main__":
    parent_dir = Path(os.path.dirname(__file__)).resolve().parents[0].as_posix()
    dataset_dir = (os.path.join(parent_dir, "tmp"),)
    model_save_path = os.path.join(parent_dir, "tmp", "yoochoose_model.data")
    logger.info(f"dataset dir -> {dataset_dir}")
    logger.info(f"model_save_path -> {model_save_path}")
    m = YooChooseGNN(
        dataset_dir=os.path.join(parent_dir, "tmp"),
        model_save_path=os.path.join(parent_dir, "tmp", "yoochoose_model.data"),
    )
    m.pipeline()
