import time
import copy
import torch
import random
import numpy as np
from torch_geometric.data import Data
from sklearn.metrics import f1_score, roc_auc_score


def test_f1(model, data, data_mask):
    """Evaluate the model on test set and print the accuracy score."""
    data.to(model.device)
    model.eval()
    _, out = model(data.x, data.edge_index)
    return f1_score(data.y[data_mask].cpu(), out.cpu().argmax(dim=1)[data_mask], average='micro')


def test_auc(model, data, data_mask):
    """Evaluate the model on test set and print the accuracy score."""
    data.to(model.device)
    model.eval()
    _, out = model(data.x, data.edge_index)
    y = data.y[data_mask].cpu()
    pred = out[data_mask].cpu()
    return roc_auc_score(y.detach().numpy(), pred.detach().numpy(), average='macro', multi_class='ovr')


def select_equal_classes(y, node_ids, num_nodes, num_classes):
    """Selects node ids from a set having equal distribution of classes"""
    selected_node_ids = []
    nodes_per_class = int(num_nodes / num_classes)

    for i in range(num_classes):
        class_node_ids = []
        for j in node_ids:
            if y[j] == i:
                class_node_ids.append(j)
        if len(class_node_ids) > nodes_per_class:
            class_node_ids = random.sample(class_node_ids, nodes_per_class)
        selected_node_ids.extend(class_node_ids)

    if len(selected_node_ids) < num_nodes:
        remaining_node_ids = list(set(node_ids) - set(selected_node_ids))
        remaining_num_nodes = num_nodes - len(selected_node_ids)
        selected_node_ids.extend(random.sample(remaining_node_ids, remaining_num_nodes))

    return selected_node_ids


def rebalance_class(class_id, class_ids, class_mask, other_mask, other_mask_2, y):
    """Rebalances a set of nodes. This means if the set contains 0 items of one class, some are taken from the other
    sets and placed in the current set"""
    if class_ids[class_id] == 0:
        other_mask_ids = []
        other_mask_2_ids = []
        for i in range(len(class_mask)):
            if other_mask[i] and y[i] == class_id:
                other_mask_ids.append(i)
            if other_mask_2[i] and y[i] == class_id:
                other_mask_2_ids.append(i)

        # Select the 10% of the class items from the other two masks and put them in the current one
        class_new_ids = random.sample(other_mask_ids, int(len(other_mask_ids) * 0.1))
        other_mask[class_new_ids] = False
        class_mask[class_new_ids] = True

        class_new_ids = random.sample(other_mask_2_ids, int(len(other_mask_2_ids) * 0.1))
        other_mask_2[class_new_ids] = False
        class_mask[class_new_ids] = True

    return class_mask, other_mask, other_mask_2


def count_nodes_per_class(y, train_mask, val_mask, test_mask):
    train_1 = 0
    train_2 = 0
    train_3 = 0

    val_1 = 0
    val_2 = 0
    val_3 = 0

    test_1 = 0
    test_2 = 0
    test_3 = 0
    for i in range(len(train_mask)):
        if train_mask[i]:
            if y[i] == 0:
                train_1 += 1
            if y[i] == 1:
                train_2 += 1
            if y[i] == 2:
                train_3 += 1
        if val_mask[i]:
            if y[i] == 0:
                val_1 += 1
            if y[i] == 1:
                val_2 += 1
            if y[i] == 2:
                val_3 += 1
        if test_mask[i]:
            if y[i] == 0:
                test_1 += 1
            if y[i] == 1:
                test_2 += 1
            if y[i] == 2:
                test_3 += 1

    return [train_1, train_2, train_3], [val_1, val_2, val_3], [test_1, test_2, test_3]


def graph_to_data_object(node_features, edge_list, train, val, test):
    """Build a PyTorch Geometric Data object from a graph and its features and split the data into train/val/test"""
    node_features = node_features.reset_index()

    # Convert party from categorical to numerical
    party = node_features['party'].to_numpy()
    y = np.full(len(node_features), fill_value=-1, dtype=int)
    for i in range(len(node_features)):
        if party[i] == 'left':
            y[i] = 0
        elif party[i] == 'middle':
            y[i] = 1
        elif party[i] == 'right':
            y[i] = 2

    # Keep only existing nodes in the edge list and reindex the remaining ids
    node_indices = node_features.index.to_numpy()
    edge_index = [[], []]
    for i in range(len(edge_list)):
        if edge_list['source_node'][i] in node_indices and edge_list['target_node'][i] in node_indices:
            edge_index[0].append(int(edge_list['source_node'][i]))
            edge_index[1].append(int(edge_list['target_node'][i]))

    for i in range(len(node_indices)):
        while i < node_indices[i]:
            node_indices -= 1
            for j in range(len(edge_index[0])):
                if edge_index[0][j] > i:
                    edge_index[0][j] -= 1
                if edge_index[1][j] > i:
                    edge_index[1][j] -= 1

    new_edge_index = copy.deepcopy(edge_index)
    new_edge_index[0].extend(edge_index[1])
    new_edge_index[1].extend(edge_index[0])
    edge_index = np.array(new_edge_index)

    # Split the nodes into train/val/test sets
    usable_node_ids = node_features.index[node_features['party'].isin(['left', 'middle', 'right'])].tolist()

    num_train = int(len(usable_node_ids) * train)
    num_val = int(len(usable_node_ids) * val)
    num_test = int(len(usable_node_ids) * test)

    train_mask = np.zeros(len(node_features), dtype=bool)
    val_mask = np.zeros(len(node_features), dtype=bool)
    test_mask = np.zeros(len(node_features), dtype=bool)

    test_ids = select_equal_classes(y, usable_node_ids, num_test, 3)
    test_mask[test_ids] = True
    usable_node_ids = list((set(usable_node_ids) - set(test_ids)))

    train_ids = select_equal_classes(y, usable_node_ids, num_train, 3)
    train_mask[train_ids] = True
    usable_node_ids = list((set(usable_node_ids) - set(train_ids)))

    val_ids = select_equal_classes(y, usable_node_ids, num_val, 3)
    val_mask[val_ids] = True

    train_ids, val_ids, test_ids = count_nodes_per_class(y, train_mask, val_mask, test_mask)

    train_mask, val_mask, test_mask = rebalance_class(0, train_ids, train_mask, val_mask, test_mask, y)
    train_mask, val_mask, test_mask = rebalance_class(1, train_ids, train_mask, val_mask, test_mask, y)
    train_mask, val_mask, test_mask = rebalance_class(2, train_ids, train_mask, val_mask, test_mask, y)

    val_mask, train_mask, test_mask = rebalance_class(0, val_ids, val_mask, train_mask, test_mask, y)
    val_mask, train_mask, test_mask = rebalance_class(1, val_ids, val_mask, train_mask, test_mask, y)
    val_mask, train_mask, test_mask = rebalance_class(2, val_ids, val_mask, train_mask, test_mask, y)

    test_mask, train_mask, val_mask = rebalance_class(0, test_ids, test_mask, train_mask, val_mask, y)
    test_mask, train_mask, val_mask = rebalance_class(1, test_ids, test_mask, train_mask, val_mask, y)
    test_mask, train_mask, val_mask = rebalance_class(2, test_ids, test_mask, train_mask, val_mask, y)

    train_ids, val_ids, test_ids = count_nodes_per_class(y, train_mask, val_mask, test_mask)

    train_n = 0
    val_n = 0
    test_n = 0
    for i in range(len(train_mask)):
        if train_mask[i]:
            train_n += 1
        if val_mask[i]:
            val_n += 1
        if test_mask[i]:
            test_n += 1

    print(f"\nTrain Nodes: {train_n}, Val Nodes: {val_n}, Test Nodes: {test_n}")

    print(f"\nTrain Set Class Distribution: Left: {train_ids[0]}, Middle: {train_ids[1]}, Right: {train_ids[2]}")
    print(f"Val Set Class Distribution: Left: {val_ids[0]}, Middle: {val_ids[1]}, Right: {val_ids[2]}")
    print(f"Test Set Class Distribution: Left: {test_ids[0]}, Middle: {test_ids[1]}, Right: {test_ids[2]}")

    x = node_features.drop(columns=['party']).astype('float64').to_numpy()

    means = x.mean()
    stds = x.std()
    normalized_data = (x - means) / stds

    x = torch.FloatTensor(normalized_data)
    edge_index = torch.LongTensor(edge_index)
    y = torch.LongTensor(y)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask,
                test_mask=test_mask)


def make_cv_splits(data, cv):
    """Splits the data into cv-splits used for Cross Validation"""
    dataset = np.zeros(len(data.train_mask), dtype=bool)
    dataset_indices = []
    num_val_og_data = 0
    num_train_og_data = 0
    for i in range(len(dataset)):
        if data.val_mask[i]:
            num_val_og_data += 1
            dataset[i] = True
            dataset_indices.append(i)
        if data.train_mask[i]:
            num_train_og_data += 1
            dataset[i] = True
            dataset_indices.append(i)

    cv_splits = []
    num_val = min(int(len(dataset_indices) / cv), num_val_og_data)
    num_train = min(int(len(dataset_indices) - num_val), num_train_og_data)
    remaining_val_ids = dataset_indices.copy()
    for i in range(cv):
        val_ids = random.sample(remaining_val_ids, num_val)
        val_mask = np.zeros(len(data.train_mask), dtype=bool)
        val_mask[val_ids] = True

        train_indices = list((set(dataset_indices) - set(val_ids)))
        train_ids = random.sample(train_indices, num_train)
        train_mask = np.zeros(len(data.train_mask), dtype=bool)
        train_mask[train_ids] = True

        cv_splits.append([train_mask, val_mask])

        remaining_val_ids = list((set(remaining_val_ids) - set(val_ids)))

    return cv_splits
