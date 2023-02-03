import networkx as nx
from statistics import mean
import pickle
import warnings
import pandas as pd
import networkx as nx
import preprocessing
import feature_engineering
warnings.filterwarnings("ignore")
from dataprep.eda import create_report

# Graph characteristics
# ----------------------

# Graph clustering coefficient
import networkx as nx
from statistics import mean

# Graph characteristics
# ----------------------

# Graph clustering coefficient
def graph_clustering_coefficient(G, df):
    cc = nx.average_clustering(G)
    df['graph_cc'] = cc

    return df


# Graph transitivity
def graph_transitivity(G, df):
    transitivity = nx.transitivity(G)
    df['graph_transitivity'] = transitivity

    return df


# Graph average degree centrality
def graph_degree_centrality(G, df):
    degree_centrality = nx.degree_centrality(G)
    df['graph_average_degree_centrality'] = mean(degree_centrality)

    return df


# Connected or not graph
def connected_graph(G, df):
    if nx.is_connected(G):
        df['is_connected'] = 1
        connected = True
    else:
        df['is_connected'] = 0
        connected = False

    return df, connected


# Strongly connected subgraph of G
def strongly_connected_subgraph(G, df):
    strongly_components = sorted(nx.connected_components(G), key=len, reverse=True)
    G_strongly = nx.subgraph(G, strongly_components[0])

    df['belongs_to_strongly_connected'] = 0
    nodes = list(G_strongly.nodes)
    for n in nodes:
        df.loc[n, ['belongs_to_strongly_connected']] = 1

    return df, G_strongly


# Graph average distance
def graph_average_distance(G, df):
    avg_d = nx.average_shortest_path_length(G)

    df['graph_average_distance'] = 0
    nodes = list(G.nodes)
    for n in nodes:
        df.loc[n, ['graph_average_distance']] = avg_d

    return df


# Graph diameter
def graph_diameter(G, df):
    diameter = nx.diameter(G)

    df['graph_diameter'] = 0
    nodes = list(G.nodes)
    for n in nodes:
        df.loc[n, ['graph_diameter']] = diameter

    return df


# Graph radius
def graph_radius(G, df):
    radius = nx.radius(G)

    df['graph_radius'] = 0
    nodes = list(G.nodes)
    for n in nodes:
        df.loc[n, ['graph_radius']] = radius

    return df


# Graph periphery
def graph_periphery(G, df):
    periphery = list(nx.periphery(G))
    df['belongs_to_graph_periphery'] = 0
    for p in periphery:
        df.loc[p, ['belongs_to_graph_periphery']] = 1

    return df


# Graph center
def graph_center(G, df):
    center = list(nx.center(G))
    df['belongs_to_graph_center'] = 0
    for p in center:
        df.loc[p, ['belongs_to_graph_center']] = 1

    return df


# Graph node connectivity
def graph_node_connectivity(G, df):
    node_connectivity = nx.node_connectivity(G)

    df['graph_node_connectivity'] = 0
    nodes = list(G.nodes)
    for n in nodes:
        df.loc[n, ['graph_node_connectivity']] = node_connectivity

    return df


# Graph edge connectivity
def graph_edge_connectivity(G, df):
    edge_connectivity = nx.edge_connectivity(G)

    df['graph_edge_connectivity'] = 0
    nodes = list(G.nodes)
    for n in nodes:
        df.loc[n, ['graph_edge_connectivity']] = edge_connectivity

    return df


# simple function to add all graph characteristics
def add_graph_characteristics(G, df):
    print("graph_clustering_coefficient")
    df = graph_clustering_coefficient(G, df)
    print("graph_transitivity")
    df = graph_transitivity(G, df)
    print("graph_degree_centrality")
    df = graph_degree_centrality(G, df)
    print("connected_graph")
    df, connected = connected_graph(G, df)

    if not connected:
        df, G = strongly_connected_subgraph(G, df)

    # the following features must be added only if the graph is connected or to the nodes of the strongly connected component
    #df = graph_average_distance(G, df)
    #df = graph_diameter(G, df)
    #df = graph_radius(G, df)
    print("graph_periphery")
    df = graph_periphery(G, df)
    print("graph_center")
    df = graph_center(G, df)
    #df = graph_node_connectivity(G, df)
    #df = graph_edge_connectivity(G, df)

    return df, G

# --------------------------------------------------------------------------------------- #

# Node characteristics
# ----------------------

# Node degree
def node_degree(G, df):
    df['node_degree'] = 0
    for node in list(G.nodes):
        df.loc[node, ['node_degree']] = G.degree(node)

    return df


# Node clustering coefficient
def node_clustering_coefficient(G, df):
    df['node_cc'] = 0
    for node in list(G.nodes):
        df.loc[node, ['node_cc']] = nx.clustering(G, node)

    return df


# Node eccentricity
def graph_eccentricity(G, df):
    eccentricity = dict(nx.eccentricity(G))

    df['graph_eccentricity'] = 0
    nodes = list(G.nodes)
    for n in nodes:
        df.loc[n, ['graph_eccentricity']] = eccentricity.get(n)

    return df


# Node degree centrality
def node_degree_centrality(G, df):
    degree_centrality = nx.degree_centrality(G)
    df['node_degree_centrality'] = 0
    for node in list(G.nodes):
        df.loc[node, ['node_degree_centrality']] = degree_centrality[node]

    return df


# Node closeness centrality
def node_closeness_centrality(G, df):
    closeness_centrality = nx.closeness_centrality(G)
    df['node_closeness_centrality'] = 0
    for node in list(G.nodes):
        df.loc[node, ['node_closeness_centrality']] = closeness_centrality[node]

    return df


# Node betweenness centrality
def node_betweenness_centrality(G, df):
    betweenness_centrality = nx.betweenness_centrality(G, normalized=True, k=int(len(list(G.nodes()))/20))
    df['node_betweenness_centrality'] = 0
    for node in list(G.nodes):
        df.loc[node, ['node_betweenness_centrality']] = betweenness_centrality[node]

    return df


# Node pagerank
def node_pageRank(G, df):
    pageRank = nx.pagerank(G, alpha=0.9)
    df['node_pageRank'] = 0
    for node in list(G.nodes):
        df.loc[node, ['node_pageRank']] = pageRank[node]

    return df


# Node hub and authority scores
def node_hits(G, df):
    hubs, authorities = nx.hits(G)
    df['node_hub'] = 0
    df['node_authority'] = 0
    for node in list(G.nodes):
        df.loc[node, ['node_hub']] = hubs[node]
        df.loc[node, ['node_authority']] = authorities[node]

    return df


# simple function to add all node characteristics
def add_node_characteristics(G, df, G_strongly):
    print("node_degree")
    df = node_degree(G, df)
    print("node_clustering_coefficient")
    df = node_clustering_coefficient(G, df)
    print("graph_eccentricity")
    df = graph_eccentricity(G_strongly, df)
    print("node_degree_centrality")
    df = node_degree_centrality(G, df)
    print("node_closeness_centrality")
    df = node_closeness_centrality(G, df)
    print("node_betweenness_centrality")
    df = node_betweenness_centrality(G, df)
    print("node_pageRank")
    df = node_pageRank(G, df)
    print("node_hits")
    df = node_hits(G, df)

    return df



G = nx.read_gpickle('graph_61')
graph_features = pd.DataFrame()
graph_features, G_strongly = add_graph_characteristics(G, graph_features)
graph_features.to_pickle('graph_features.pkl')
node_features = pd.DataFrame()
node_features = add_node_characteristics(G, node_features, G_strongly)
node_features.to_pickle('node_features.pkl')