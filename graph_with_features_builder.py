import pandas as pd
import networkx as nx
"""
input:  
        networkx Graph: 'G'
        dataframe with node features : 'node_features'
        custom size for the output graph, -1 -> full graph : 'graph_node_size'
        
return : 

        A networkx Graph with node features

exception : 

        The size of the nodes of the graph 'G' must me the same with the size of the
        rows of the dataframe 'node_features
            
"""
def graph_with_node_attributes(G,node_features,graph_node_size=-1):

    nodes = list(range(0,G.number_of_nodes()))
    attrs_list = node_features.to_dict(orient='records')
    attrs_dict = {}
    if graph_node_size == -1:
        pass
    elif graph_node_size > max(len(nodes),len(attrs_list)):
        raise Exception("graph_node_size exceeds the limits of the given graph or the node features")
    else:
        nodes = nodes[0:graph_node_size]
        attrs_list = attrs_list[0:graph_node_size]
        G = G.subgraph(nodes)

    if len(nodes) != len(attrs_list):
        raise Exception("Graph size : " + str(len(nodes)) + " - Node Features size: " + str(len(attrs_list)) + " - The sizes must me the same.")

    for node_features,node in zip(attrs_list,nodes):
        attrs_dict[node] = node_features

    nx.set_node_attributes(G, attrs_dict)
    print(G) # Graph with X nodes and Y edges
    return G


# test code
# G = pd.read_pickle('data/graph')
# graph_dict = pd.read_pickle('graph_df')
# G_with_node_features = graph_with_node_attributes(G,graph_dict,5)
# print(G_with_node_features.nodes.data())
