import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils.convert import to_networkx, from_networkx
from graph_with_features_builder import graph_with_node_attributes
import pandas as pd
from utils import *


class GAT(torch.nn.Module):
    """Graph Attentional Network"""

    def __init__(self, config, dim_in, dim_out):
        super().__init__()
        if config['architecture'] == 'two-layer':
            self.gat1 = GATConv(dim_in, config['dim_h1'])
            self.gat2 = GATConv(config['dim_h1'], dim_out)
        else:
            self.gat1 = GATConv(dim_in, config['dim_h1'])
            self.gat2 = GATConv(config['dim_h1'], config['dim_h2'])
            self.gat3 = GATConv(config['dim_h2'], dim_out)

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=config['lr'],
                                          weight_decay=config['l2'])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.config = config

    def forward(self, x, edge_index):
        if self.config['architecture'] == 'two-layer':
            h = self.gat1(x, edge_index)
            h = F.relu(h)
            h = self.gat2(h, edge_index)
            return h, F.softmax(h, dim=1)
        else:
            h = self.gat1(x, edge_index)
            h = F.relu(h)
            h = self.gat2(h, edge_index)
            h = F.relu(h)
            h = self.gat3(h, edge_index)
            return h, F.softmax(h, dim=1)

    def fit(self, data, stats):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = self.optimizer
        data.to(self.device)

        self.train()
        for epoch in range(self.config['epochs'] + 1):
            # Training
            optimizer.zero_grad()
            _, out = self(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            f1 = f1_score(data.y[data.train_mask].cpu(), out[data.train_mask].cpu().argmax(dim=1), average='micro')
            loss.backward()
            optimizer.step()

            # Print metrics every 10 epochs
            if epoch % 10 == 0 and loss is not None and stats:
                print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} '
                      f'| Train F1: {f1}:.2f')


def grid_search_cv(data, cv, param_grid):
    """Performs Grid Search for the GAT model and keeps the best performing model based on the F1 score"""
    cv_splits = make_cv_splits(data, cv)

    best_model = None
    best_config = None
    best_model_f1 = 0
    for i in range(len(param_grid)):
        for arch in param_grid[i]['architecture']:
            for lr in param_grid[i]['lr']:
                for l2 in param_grid[i]['l2']:
                    for epochs in param_grid[i]['epochs']:
                        for dim_h1 in param_grid[i]['dim_h1']:
                            for dim_h2 in param_grid[i]['dim_h2']:
                                config = {
                                    'architecture': arch,
                                    'lr': lr,
                                    'l2': l2,
                                    'epochs': epochs,
                                    'dim_h1': dim_h1,
                                    'dim_h2': dim_h2
                                }
                                start_time = time.time()
                                f1_sum = 0
                                for cv_split in cv_splits:
                                    cv_data = Data(x=data.x, edge_index=data.edge_index, y=data.y,
                                                   train_mask=cv_split[0], val_mask=cv_split[1],
                                                   test_mask=data.test_mask)
                                    model = GAT(config, data.num_features, 3)
                                    model.fit(cv_data, False)
                                    model_f1 = test_f1(model, cv_data, cv_data.val_mask)

                                    f1_sum += model_f1
                                avg_f1 = f1_sum / len(cv_splits)
                                print(f"Config: {config}, Run time: {time.time() - start_time:.0f} seconds")

                                if avg_f1 > best_model_f1:
                                    best_model_f1 = avg_f1
                                    best_model = model
                                    best_config = config

    return best_model, best_config
