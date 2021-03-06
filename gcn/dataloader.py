import os

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from torch_geometric.data import InMemoryDataset, Data
from sklearn.preprocessing import LabelEncoder

TMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tmp"))
click_df_path = os.path.join(TMP_DIR, "dataset/yoochoose-clicks.dat")
buy_df_path = os.path.join(TMP_DIR, "dataset/yoochoose-buys.dat")


class YooChooseDataset(InMemoryDataset):
    def __init__(
        self, root, transform=None, pre_transform=None, pre_filter=None,
    ):
        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )
        self.data, self.slices, self.num_embeddings = torch.load(
            self.processed_paths[0]
        )

    @property
    def raw_file_names(self):
        logger.info("Required files check!")
        return [
            os.path.join(TMP_DIR, "dataset/clicks_small.dat"),
            os.path.join(TMP_DIR, "dataset/buys_small.dat"),
        ]

    @property
    def processed_file_names(self):
        return [os.path.join(TMP_DIR, "yoochoose_click_binary_1M_sess.dataset")]

    def download(self):
        logger.error(
            f"Can not find files {self.raw_paths}. Please check and place the files in correct path!"
        )
        raise FileNotFoundError

    def initialize_df(self, sample_sessions=None):
        logger.info("Loading dataset...")
        self.df = pd.read_csv(click_df_path, header=None)
        self.df.columns = ["session_id", "timestamp", "item_id", "category"]

        # remove sessions with less than 3 item ids
        logger.info("Removing sessions with less than 3 item ids...")
        self.df["valid_session"] = self.df.session_id.map(
            self.df.groupby("session_id")["item_id"].size() > 2
        )
        self.df = self.df.loc[self.df.valid_session].drop("valid_session", axis=1)

        # the number of sessions might be very high, performing sampling
        if sample_sessions:
            logger.info(f"Sampling {sample_sessions} sessions...")
            sampled_session_id = np.random.choice(
                self.df.session_id.unique(), sample_sessions, replace=False
            )
            self.df = self.df.loc[self.df.session_id.isin(sampled_session_id)].copy()

        # map the item ids to a small range
        logger.info("Mapping item ids to smaller range...")
        item_encoder = LabelEncoder()
        self.df["item_id"] = item_encoder.fit_transform(self.df.item_id)

        # get the target class (buy or not buy event) for the sessions
        logger.info("Determining the target class of the sessions (buy/not buy)...")
        buy_df = pd.read_csv(buy_df_path, header=None)
        buy_df.columns = ["session_id", "timestamp", "item_id", "price", "quantity"]
        self.df["label"] = self.df.session_id.isin(buy_df.session_id)
        self.num_embeddings = self.df.item_id.max() + 1
        del buy_df
        logger.info("Loading dataset done!")

    def process(self):
        self.initialize_df(sample_sessions=250000)
        data_list = []

        logger.info("Processing dataset...")
        grouped = self.df.groupby("session_id")
        for session_id, group in tqdm(grouped):
            sess_item_id = LabelEncoder().fit_transform(group.item_id)
            group = group.reset_index(drop=True)
            group["sess_item_id"] = sess_item_id
            node_features = (
                group.loc[group.session_id == session_id, ["sess_item_id", "item_id"]]
                .sort_values("sess_item_id")
                .item_id.drop_duplicates()
                .values
            )

            node_features = torch.LongTensor(node_features).unsqueeze(1)
            target_nodes = group.sess_item_id.values[1:]
            source_nodes = group.sess_item_id.values[:-1]

            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            x = node_features
            # the session id is same for all rows in a session (whether buy event or not)
            y = torch.FloatTensor([group.label.values[0]])

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        logger.info("Completed processing!")
        data, slices = self.collate(data_list)
        torch.save((data, slices, self.num_embeddings), self.processed_paths[0])
        logger.info(f"Processed files saved as {self.processed_paths[0]}")


if __name__ == "__main__":
    o = YooChooseDataset(root=TMP_DIR)
    print(o.data)
