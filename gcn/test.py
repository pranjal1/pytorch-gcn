# import os

# import networkx as nx
# import torch
# import numpy as np
# import pandas as pd
# from torch_geometric.datasets import Planetoid
# from torch_geometric.utils.convert import to_networkx


# TMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tmp"))

# dataset1 = Planetoid(root=TMP_DIR, name="Cora")

# cora = dataset1[0]
# coragraph = to_networkx(cora)

# node_labels = cora.y[list(coragraph.nodes)].numpy()

# import matplotlib.pyplot as plt

# # plt.figure(1,figsize=(14,12))
# nx.draw(
#     coragraph,
#     cmap=plt.get_cmap("Set1"),
#     node_color=node_labels,
#     node_size=25,
#     linewidths=10,
# )
# plt.savefig(os.path.join(TMP_DIR, "nw.png"))
