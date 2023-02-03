# PyTorch
import torch
import torch.nn.functional as F

# PyTorch Geometric
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import f1_score

from utils import *


class GraphSAGEGCN(torch.nn.Module):
    """GraphSAGEGCN"""

    def __init__(self, config, dim_in, dim_out):
        super().__init__()
        self.sage = SAGEConv(dim_in, config['dim_h1'], aggr=config['aggr'], project=config['proj'])
        self.gcn = GCNConv(config['dim_h1'], dim_out)

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=config['lr'],
                                          weight_decay=config['l2'])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.config = config

    def forward(self, x, edge_index):
        h = self.sage(x, edge_index)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.gcn(h, edge_index)
        return h, F.softmax(h, dim=1)

    def fit(self, data, stats):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = self.optimizer
        data.to(self.device)
        input_nodes = []
        for i in range(len(data.train_mask)):
            if data.train_mask[i]:
                input_nodes.append(i)
        input_nodes = torch.LongTensor(input_nodes)
        train_loader = NeighborLoader(
            data,
            num_neighbors=[10, 10],
            batch_size=self.config['batch'],
            input_nodes=input_nodes,
        )

        self.train()
        for epoch in range(1, self.config['epochs'] + 1):
            f1 = 0
            loss = None
            # Train on batches
            for batch in train_loader:
                optimizer.zero_grad()
                _, out = self(batch.x, batch.edge_index)
                loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
                f1 += f1_score(out[batch.train_mask].cpu().argmax(dim=1), batch.y[batch.train_mask].cpu(),
                               average='micro')
                loss.backward()
                optimizer.step()

            # Print metrics every 10 epochs
            if epoch % 10 == 0 and loss is not None and stats:
                print(f'Epoch {epoch:>3} | Train Loss: {loss / len(train_loader):.3f} '
                      f'| Train F1: {f1 / len(train_loader):.2f}')


def grid_search_cv(data, cv, param_grid):
    """Performs Grid Search for the GraphSAGE-GCN model and keeps the best performing model based on the F1 score"""
    cv_splits = make_cv_splits(data, cv)

    best_model = None
    best_config = None
    best_model_f1 = 0
    for i in range(len(param_grid)):
        for batch in param_grid[i]['batch']:
            for lr in param_grid[i]['lr']:
                for l2 in param_grid[i]['l2']:
                    for aggr in param_grid[i]['aggr']:
                        for proj in param_grid[i]['proj']:
                            for epochs in param_grid[i]['epochs']:
                                for dim_h1 in param_grid[i]['dim_h1']:
                                    config = {
                                        'batch': batch,
                                        'lr': lr,
                                        'l2': l2,
                                        'aggr': aggr,
                                        'proj': proj,
                                        'epochs': epochs,
                                        'dim_h1': dim_h1,
                                    }
                                    start_time = time.time()
                                    f1_sum = 0
                                    for cv_split in cv_splits:
                                        cv_data = Data(x=data.x, edge_index=data.edge_index, y=data.y,
                                                       train_mask=cv_split[0], val_mask=cv_split[1],
                                                       test_mask=data.test_mask)
                                        model = GraphSAGEGCN(config, data.num_features, 3)
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
