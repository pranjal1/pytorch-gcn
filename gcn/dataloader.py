import os

import torch
from tqdm import tqdm
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data
from sklearn.preprocessing import LabelEncoder

TMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tmp"))


class YooChooseDataset(InMemoryDataset):
    def __init__(
        self,
        click_df_path,
        buy_df_path,
        root=None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )
        self.click_df_path = click_df_path
        self.buy_df_path = buy_df_path
        self.initialize_df()
        self.data, self.slices = torch.load(self.processed_paths[0])

    def initialize_df(self):
        self.df = pd.read_csv(self.click_df_path, header=None)
        self.df.columns = ["session_id", "timestamp", "item_id", "category"]
        buy_df = pd.read_csv(self.buy_df_path, header=None)
        buy_df.columns = ["session_id", "timestamp", "item_id", "price", "quantity"]
        self.df["label"] = self.df.session_id.isin(buy_df.session_id)
        del buy_df

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [os.path.join(TMP_DIR, "yoochoose_processed.dataset")]

    def download(self):
        pass

    def process(self):

        data_list = []

        # process by session_id
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

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
