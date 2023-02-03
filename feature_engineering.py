import networkx as nx
from statistics import mean

# Graph characteristics
# ----------------------

# Graph clustering coefficient
def graph_clustering_coefficient(G, df):
    cc = nx.average_clustering(G)
    nodes = list(G.nodes)
    for n in nodes:
        df.loc[n, ['graph_cc']] = cc

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


# simple function to add all graph characteristics
def add_graph_characteristics(G, df):
    df = graph_clustering_coefficient(G, df)
    df = graph_transitivity(G, df)
    df = graph_degree_centrality(G, df)
    df, connected = connected_graph(G, df)

    if not connected:
        df, G = strongly_connected_subgraph(G, df)

    # the following features must be added only if the graph is connected or to the nodes of the strongly connected component
    print("in periphery")
    df = graph_periphery(G, df)
    print("in center")
    df = graph_center(G, df)

    # returns the updated (strongly) G
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
    df = node_degree(G, df)
    df = node_clustering_coefficient(G, df)
    print("in eccentricity")
    df = graph_eccentricity(G_strongly, df)
    df = node_degree_centrality(G, df)
    print("in closeness_centrality")
    df = node_closeness_centrality(G, df)
    print("in betweenness_centrality")
    df = node_betweenness_centrality(G, df)
    df = node_pageRank(G, df)
    df = node_hits(G, df)

    return df

# --------------------------------------------------------------------------------------- #
