
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

def plot_graph_with_similarities(data, similarity_matrix, node_labels):

    # Convert PyTorch Geometric data to NetworkX graph
    G = to_networkx(data, to_undirected=True)

    # Extract edge indices from the graph
    edges = list(G.edges())

    # Compute weights for the edges based on similarity_matrix
    weights = [similarity_matrix[u, v] for u, v in edges]

    # Normalize weights for visualization (scale between 0.5 and 3 for line thickness)
    min_weight, max_weight = min(weights), max(weights)
    normalized_weights = [
        0.5 + 2.5 * (w - min_weight) / (max_weight - min_weight) for w in weights
    ]

    # Node colors based on classification labels
    node_colors = node_labels.numpy()
    color_map = plt.cm.get_cmap('viridis', len(set(node_colors)))

    # Plot the graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)  # Layout for positioning nodes

    # Draw edges with varying thickness
    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.5,
        width=normalized_weights
    )

    # Draw nodes with color coding
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        cmap=color_map,
        node_size=100
    )

    # Add a color bar
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(), label="Node Classification Labels")  # Explicitly link to current axes

    plt.title("Graph Visualization with Nodes Color-Coded by Labels and Weighted Edges")
    plt.axis('off')

    # Save the graph visualization
    output_file = "graph_with_similarities.png"
    plt.savefig(output_file)
    plt.close()
    return output_file




def graph_creation(all_embeddings, all_labels, train_mask, val_mask, test_mask, all_idx):
    # Define the cosine similarity threshold
    
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(all_embeddings)

    unique_values = np.unique(similarity_matrix)

    # Print the number of unique values and optionally a sample
    print(f"Number of unique values: {len(unique_values)}")


    plt.figure(figsize=(8, 6))
    plt.boxplot(unique_values, vert=False)
    plt.title("Box Plot of Unique Values in Similarity Matrix")
    plt.xlabel("Similarity Score")
    plt.ylabel("Values")
    plt.grid(True)
    

    # Save the figure to a file
    output_file = "sim_fig.png"
    plt.savefig(output_file)
    plt.close()

    similarity_threshold = similarity_matrix.mean()  # Adjust based on your requirements
    print(similarity_threshold)
    # Create edges based on the threshold
    edges = []
    for i in range(len(all_embeddings)):
        for j in range(len(all_embeddings)):
            if i != j and similarity_matrix[i, j] >= similarity_threshold:  # Avoid self-loops
                edges.append((i, j))

    # Convert edges to a PyTorch tensor
    edge_index = torch.tensor(edges, dtype=torch.long).t()

    # Convert features and labels to tensors
    x = torch.tensor(all_embeddings, dtype=torch.float)
    y = torch.tensor(all_labels, dtype=torch.long)

    # Create the graph data object
    data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    data.idx = all_idx
    #plot_graph_with_similarities(data, similarity_matrix, y)

    # Print some information about the graph
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")

    return data,x,y, all_idx







def graph_preprocessing(folder):

    "extract the embeddings given folder"

    train,val,test=data_preprocessing(folder, emb_length=128)

    all_embeddings = np.vstack([train['emb'], val['emb'], test['emb']])
    all_labels = np.hstack([train['label'], val['label'], test['label']])
    all_idx = np.hstack([train['idx'], val['idx'], test['idx']])
    

    # Create masks
    train_mask = torch.zeros(len(all_embeddings), dtype=torch.bool)
    train_mask[:len(train['emb'])] = True

    val_mask = torch.zeros(len(all_embeddings), dtype=torch.bool)
    val_mask[len(train['emb']):len(train['emb']) + len(val['emb'])] = True

    test_mask = torch.zeros(len(all_embeddings), dtype=torch.bool)
    test_mask[-len(test['emb']):] = True

    return graph_creation(all_embeddings, all_labels, train_mask, val_mask, test_mask, all_idx)



def data_preprocessing(folder, emb_length):
    '''
    Given a folder from runs/ , we will get the train, val, and test embedding

    '''
    train=pd.read_csv(folder+"/training_emb.csv")
    val=pd.read_csv(folder+"/validation_emb.csv")
    test=pd.read_csv(folder+"/testing_emb.csv")

    print(train)
    train_embedding, val_embedding, test_embedding = train.iloc[:,:emb_length], val.iloc[:,:emb_length], test.iloc[:,:emb_length]
    train_label, val_label, test_label = train['label'].values, val['label'].values, test['label'].values
    train_idx, val_idx, test_idx = train['idx'].values, val['idx'].values, test['idx'].values


    train={'emb':train_embedding, 'label':train_label, 'idx':train_idx}
    val={'emb':val_embedding, 'label':val_label, 'idx':val_idx}
    test={'emb':test_embedding, 'label':test_label, 'idx':test_idx}
    

    return train,val,test


    

