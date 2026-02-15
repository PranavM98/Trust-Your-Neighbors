import faiss
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from sklearn.metrics import confusion_matrix
from collections import Counter
import numpy as np
from sklearn.metrics import roc_curve

import time



'''

Weighted average of K neighbor embedding 



'''
# ----------------------------
# Set Device
# ----------------------------
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")


# ----------------------------
# Load CSV Data
# ----------------------------
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    embeddings = data.iloc[:, :128].values  # First 128 columns: embeddings
    labels = data['label'].values          # Class labels
    indices = data['idx'].values           # Accession indices
    return embeddings, labels, indices

# ----------------------------
# Build Neighbors Search
# ----------------------------
def build_faiss_index(embeddings):
    """
    Build an FAISS index for dot-product search (IndexFlatIP).
    If you'd rather have Euclidean distances, use IndexFlatL2.
    """
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Cosine or dot-product similarity
    index.add(embeddings.astype(np.float32))
    return index

def get_neighbors(index, query_embeddings, K):
    """
    Returns 'distances' and 'indices' of the K nearest neighbors
    using the given FAISS index.
    Note: 'distances' here are dot-product similarities if using IndexFlatIP.
    """
    distances, idxs = index.search(query_embeddings.astype(np.float32), K)
    return distances, idxs

# ----------------------------
# Dataset and DataLoader
# ----------------------------
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# ----------------------------
# Prepare Batch Features
# ----------------------------
def prepare_batch_features(batch_embeddings, 
                           train_embeddings, 
                           train_labels, 
                           index, 
                           K,
                           output_dim=1):
    """
    For each embedding in the batch, find its K nearest neighbors, compute
    the weighted average of neighbor embeddings, and the weighted average
    of the neighbor label values (for binary classification, these are 0 or 1).
    Then concatenate:
       [query_embedding, weighted_neighbor_embedding, weighted_neighbor_label_avg]
    """
    
    # 1) Get the top-K neighbors for each sample in the batch

    distances, neighbor_indices = get_neighbors(index, batch_embeddings, K)

    batch_features = []
    for i, query_embedding in enumerate(batch_embeddings):
        # 2) Extract neighbor embeddings and labels
        neighbors_emb = train_embeddings[neighbor_indices[i]]  # Shape: (K, emb_dim)
        neighbors_lbl = train_labels[neighbor_indices[i]]      # Shape: (K,)
        
        # Instead of one-hot encoding, just convert neighbor labels to float
        neighbors_lbl_scalar = neighbors_lbl.astype(float)  # Shape: (K,)
        # Convert to 2D vectors (each element is now a vector of shape (2,))
        neighbor_lbl_vector = np.stack((1 - neighbors_lbl_scalar, neighbors_lbl_scalar), axis=1)



        # 4) Convert FAISS “distances” to weights
        weights = distances[i]  # shape (K, )

        exp_scores = np.exp(weights)
        weights = exp_scores / (np.sum(exp_scores) + 1e-8)

        
        # print(sim)
        # if np.all(sim <= 0):
        #     weights = np.ones_like(sim) / len(sim)
        # else:
        #     weights = sim / (sim.sum() + 1e-9)
        # print("weight:",weights)
        # 5) Compute weighted averages
        weighted_neighbor_emb = np.sum(neighbors_emb * weights[:, None], axis=0) /np.sum(weights) # shape (emb_dim,)
        # Here, compute the weighted average of the scalar labels

        # Compute weighted average
        weighted_neighbor_label_avg = np.sum(neighbor_lbl_vector * weights[:, None], axis=0) / np.sum(weights)



        # 6) Build the final feature for MLP input
        #    [ query_embedding, weighted_neighbor_emb, weighted_neighbor_label_avg ]
        input_features = np.concatenate([
            query_embedding,
            weighted_neighbor_emb,
            weighted_neighbor_label_avg  # wrap scalar in a list to concatenate
        ])

        batch_features.append(input_features)

    return np.array(batch_features)

# ----------------------------
# Compute AUROC
# ----------------------------
def compute_auroc(y_true, y_pred, num_classes):
    """
    Compute one-vs-rest AUROC for multi-class classification.
    """
    auroc_positive = roc_auc_score(y_true, y_pred)

    # Compute AUROC for the negative class (0) if needed (1 - positive AUROC)
    auroc_negative = roc_auc_score(1 - y_true, 1 - y_pred)

    print("Class 0:",auroc_negative)
    print("Class 1:", auroc_positive)

    # y_true = y_true.astype(int)
    # y_true_one_hot = np.eye(num_classes)[y_true]

    # auroc = roc_auc_score(y_true_one_hot, y_pred, average=average, multi_class="ovr")
    # #auroc = 0.0  # Handle case where AUROC can't be calculated
    return (auroc_positive+auroc_negative)/2, auroc_negative, auroc_positive

# ----------------------------
# Model Definition
# ----------------------------
class RetrievalAugmentedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Example simple MLP:
           fc1 -> ReLU -> fc2 -> Softmax
        """
        super(RetrievalAugmentedMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.4)

    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x) 
        x = self.fc3(x)
        return x

# ----------------------------
# Combine Datasets
# ----------------------------
def combine_datasets(train_embeddings, train_labels, train_indices,
                     val_embeddings, val_labels, val_indices):
    combined_embeddings = np.concatenate([train_embeddings, val_embeddings], axis=0)
    combined_labels = np.concatenate([train_labels, val_labels], axis=0)
    combined_indices = np.concatenate([train_indices, val_indices], axis=0)  # Keep ACC_NUMs
    return combined_embeddings, combined_labels, combined_indices



def find_best_threshold(y_true, y_probs):
    """
    Finds the best threshold using the ROC curve and Youden’s J statistic.
    
    Args:
        y_true (array-like): Ground truth labels (0 or 1).
        y_probs (array-like): Predicted probabilities.
    
    Returns:
        best_threshold (float): The optimal threshold.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    youden_j = tpr - fpr  # Compute Youden's J statistic
    best_index = np.argmax(youden_j)  # Get the index of max J statistic
    best_threshold = thresholds[best_index]
    
    return best_threshold



# ----------------------------
# Evaluate on Test Data
# ----------------------------
def evaluate_on_test(model, 
                     test_embeddings, 
                     test_labels, 
                     combined_embeddings, 
                     combined_labels, 
                     combined_index, 
                     K, 
                     neighbor_k,
                     output_dim=1,
                     test_indices=None,
                     combined_indices=None):
    """
    Evaluate the model on test data, and also return the top-N neighbors (ACC_NUMs).
    """
    model.eval()
    with torch.no_grad():
 
        batch_features = prepare_batch_features(
            test_embeddings, 
            combined_embeddings, 
            combined_labels, 
            combined_index, 
            K,
            output_dim=1
        )

        batch_features_tensor = torch.tensor(batch_features, dtype=torch.float32).to(device)
   
        outputs = model(batch_features_tensor)

    probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()
    test_auroc, auroc_0, auroc_1 = compute_auroc(test_labels.astype(int), probabilities, output_dim)
    best_thresh = find_best_threshold(test_labels.astype(int), probabilities)
    predicted_labels = (probabilities >= best_thresh).astype(int)
    f1 = f1_score(test_labels.astype(int), predicted_labels)
    sens = recall_score(test_labels.astype(int), predicted_labels)

    print(f"Optimal Threshold: {best_thresh:.4f}")
    print("F1 SCORE:", f1)
    print("Sensitivity:", sens)

    # ---- Get top-N neighbors (map to ACC_NUMs) ----
    _, neighbor_indices = get_neighbors(combined_index, test_embeddings, neighbor_k)
    neighbor_accnums = combined_indices[neighbor_indices]  # map to ACC_NUMs
    print(neighbor_accnums)
    neighbor_df = pd.DataFrame(
        neighbor_accnums,
        columns=[f"Neighbor_{i+1}" for i in range(neighbor_k)]
    )

    # Build results dataframe
    df_results = pd.DataFrame({
        "Index": test_indices,
        "Predicted_Disposition": predicted_labels,
        "Actual_Label": test_labels.astype(int),
        "Probability": probabilities
    })

    df_results = pd.concat([df_results, neighbor_df], axis=1)

    return test_auroc, auroc_0, auroc_1, probabilities, f1, sens, df_results


# ----------------------------
# Main Function
# ----------------------------
def rac_joint_main(folder, k):
    """
    Example main function showing:
      1) Load train/val/test data from CSV
      2) Train MLP with retrieval-augmented features
      3) Evaluate best model on test set
    """

    device = torch.device("cpu")
    # --- Load CSVs ---
    train_embeddings, train_labels, train_indices = load_data(folder + "training_emb.csv")
    val_embeddings, val_labels, val_indices     = load_data(folder + "validation_emb.csv")
    test_embeddings, test_labels, test_indices  = load_data(folder + "testing_emb.csv")

    # Create Datasets and DataLoaders
    train_dataset = EmbeddingDataset(train_embeddings, train_labels)
    val_dataset   = EmbeddingDataset(val_embeddings, val_labels)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    # --- Model Parameters ---
    K = k              # Number of neighbors
    embedding_dim = 128
    hidden_dim    = 64
    output_dim    = 1   # 3 classes

    # Because we now do: [query_emb + weighted_neighbor_emb + weighted_neighbor_label_dist],
    #   => total input_dim = 128 + 128 + 4 = 260
    # (Adjust accordingly if you alter the code.)
    input_dim = embedding_dim + embedding_dim + output_dim+1


    # --- Define model, loss, optimizer ---
    model = RetrievalAugmentedMLP(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00005)

    # Build FAISS index for training embeddings
    faiss_index = build_faiss_index(train_embeddings)

    # --- Training Loop ---
    best_val_auroc = 0.0
    best_epoch  = -1
    best_model_state = None

    epochs = 50  # Increase if needed
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_true  = []

        for batch_emb, batch_lbl in train_loader:
            batch_emb_np = batch_emb.numpy()  # (B, 128)

            # Prepare retrieval-augmented features
            batch_features = prepare_batch_features(
                batch_emb_np,
                train_embeddings,
                train_labels,
                faiss_index,
                K,
                output_dim=output_dim
            )
            
            batch_features_tensor = torch.tensor(batch_features, dtype=torch.float32).to(device)
            batch_lbl_tensor      = torch.tensor(batch_lbl, dtype=torch.float32).to(device)
            batch_lbl_tensor = batch_lbl_tensor.unsqueeze(1) 
            # Forward pass
            outputs = model(batch_features_tensor)

            loss = criterion(outputs, batch_lbl_tensor)
            

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track training stats
            train_loss += loss.item()
            train_preds.append(outputs.detach().cpu().numpy())
            train_true.append(batch_lbl.cpu().numpy())

        train_preds = np.concatenate(train_preds, axis=0)  # (N, 4)
        train_probabilities = 1 / (1 + np.exp(-train_preds))

        train_true  = np.concatenate(train_true, axis=0)   # (N, )
        train_auroc, auroc_0, auroc_1 = compute_auroc(train_true.astype(int), train_probabilities, output_dim)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true  = []

        with torch.no_grad():
            for batch_emb, batch_lbl in val_loader:
                batch_emb_np = batch_emb.numpy()

                batch_features = prepare_batch_features(
                    batch_emb_np,
                    train_embeddings,
                    train_labels,
                    faiss_index,
                    K,
                    output_dim=output_dim
                )

                batch_features_tensor = torch.tensor(batch_features, dtype=torch.float32).to(device)
                batch_lbl_tensor      = torch.tensor(batch_lbl, dtype=torch.float32).to(device)
                batch_lbl_tensor = batch_lbl_tensor.unsqueeze(1) 

                outputs = model(batch_features_tensor)
                loss = criterion(outputs, batch_lbl_tensor)

                val_loss += loss.item()
                val_preds.append(outputs.cpu().numpy())
                val_true.append(batch_lbl.cpu().numpy())

        val_preds = np.concatenate(val_preds, axis=0)
        val_probabilities = 1 / (1 + np.exp(-val_preds))

        val_true  = np.concatenate(val_true,  axis=0)
        val_auroc,auroc_0, auroc_1  = compute_auroc(val_true.astype(int), val_probabilities, output_dim)



        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Train AUC: {train_auroc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val AUC: {val_auroc:.4f}")


        # Check for best model
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch  = epoch + 1
            best_model_state = model.state_dict()


    # ----------------------------
    # Retrain on Combined Data
    # ----------------------------
    combined_embeddings, combined_labels, combined_indices = combine_datasets(
        train_embeddings, train_labels, train_indices,
        val_embeddings,   val_labels,   val_indices
    )
    combined_index = build_faiss_index(combined_embeddings)

    # Create DataLoader for combined data
    combined_dataset = EmbeddingDataset(combined_embeddings, combined_labels)
    combined_loader  = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    # Load best model
    model.load_state_dict(best_model_state)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    
    print("Retraining on combined train+val dataset...")
    for epoch in range(best_epoch):
        model.train()
        epoch_loss = 0.0
        for batch_emb, batch_lbl in combined_loader:
            batch_emb_np = batch_emb.numpy()

            batch_features = prepare_batch_features(
                batch_emb_np,
                combined_embeddings,
                combined_labels,
                combined_index,
                K,
                output_dim=output_dim
            )

            batch_features_tensor = torch.tensor(batch_features, dtype=torch.float32).to(device)
            batch_lbl_tensor      = torch.tensor(batch_lbl, dtype=torch.float32).to(device)
            batch_lbl_tensor = batch_lbl_tensor.unsqueeze(1) 

            outputs = model(batch_features_tensor)
            loss = criterion(outputs, batch_lbl_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Retrain Epoch {epoch+1}/{best_epoch} | Loss: {epoch_loss:.4f}")

    # ----------------------------
    # Evaluate on Test Data
    # ----------------------------
    return evaluate_on_test(
        model,
        test_embeddings, 
        test_labels,
        combined_embeddings,
        combined_labels,
        combined_index,
        K=k,
        neighbor_k=k,
        output_dim=output_dim,
        test_indices=test_indices,
        combined_indices=combined_indices,


    )
