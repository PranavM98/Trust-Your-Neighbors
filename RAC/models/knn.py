import torch
from torch.utils.data import Dataset
from monai.transforms import Compose, LoadImage, Resize, ScaleIntensity
from monai.config import DtypeLike
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import random
# from dataloaders.CT_dataset import NiftiDataset
from sklearn.preprocessing import LabelEncoder
from transformers import BertModel, BertConfig, BertTokenizer
# from dataloaders.Tabular_Dataset import TabularDataset
import os
import numpy as np
from sklearn.impute import KNNImputer
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
random_state=random.randint(0,100)
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

import os
import re
import pandas as pd
import os

def find_subfolder_with_prefix(folder_path, prefix):
    """
    Find a subfolder inside the given folder that starts with the specified prefix.
    
    Args:
        folder_path (str): Path to the parent folder.
        prefix (str): Prefix string to match subfolder names.
    
    Returns:
        str: Path to the first subfolder that matches the prefix or None if no match is found.
    """
    try:
        subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        for subfolder in subfolders:
            if subfolder.startswith(prefix):
                return os.path.join(folder_path, subfolder)
        return None
    except Exception as e:
        return f"Error: {e}"


def get_last_epoch_data(parent_folder):
    """
    From the given parent_folder, find all subfolders starting with `Epoch_`,
    determine the last epoch folder, and return the paths to the CSV files.
    """
    # List all entries in the parent folder that start with "Epoch_"
    epoch_folders = [d for d in os.listdir(parent_folder) if d.startswith("Epoch_")]
    
    # Extract the numeric portion of the folder name using a regular expression
    epoch_numbers = []
    for folder_name in epoch_folders:
        match = re.match(r"Epoch_(\d+)", folder_name)
        if match:
            epoch_numbers.append(int(match.group(1)))
    
    if not epoch_numbers:
        raise ValueError(f"No Epoch_ subfolders found in {parent_folder}")

    # Identify the maximum (last) epoch number
    last_epoch_num = max(epoch_numbers)
    
    # Construct the path to the last epoch folder
    last_epoch_folder = os.path.join(parent_folder, f"Epoch_{last_epoch_num}/")
    return last_epoch_folder
# ----------------------------


# ----------------------------
# Load CSV Data
# ----------------------------
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    embeddings = data.iloc[:, :128].values  # First 128 columns: embeddings
    labels = data['label'].values  # Class labels
    indices = data['idx'].values  # Accession indices
    return embeddings, labels, indices

# Load train, val, and test embeddings
# given image_model loss loss _input
folder='folder with model runs'

#change the filepaths form temporal to institutional 

results = []
for mmodel in ['metatransformer']:
    for image_model in ['ctnet','ctclip','3dresnet']:
        for input in ['label']:
            for t in ['100']:
                trial=t

                file_path=folder+mmodel+'/'+image_model+"_supcon_original_label_"+f"{trial}"
                final_folder=get_last_epoch_data(file_path)
                train_embeddings, train_labels, train_indices = load_data(final_folder + "/training_emb.csv")
                val_embeddings, val_labels, val_indices = load_data(final_folder + "/validation_emb.csv")
                test_embeddings, test_labels, test_indices = load_data(final_folder + "/testing_emb.csv")
                
                
                
                # Combine train and validation data
                combined_embeddings = np.vstack((train_embeddings, val_embeddings))
                combined_labels = np.hstack((train_labels, val_labels))
                
                # Fit KNN model
                K = 7  # Number of neighbors
                model = KNeighborsClassifier(n_neighbors=K, metric='cosine')
                model.fit(combined_embeddings, combined_labels)
                
                
                
                
                # # Make predictions on the test set
                y_pred = model.predict(test_embeddings)
                print(pd.Series(y_pred).value_counts())
                
                # Assuming you have K = 7 from before and the KNeighborsClassifier is fitted as `model`.
                # Get the indices of the K nearest neighbors for each test embedding
                distances, neighbor_indices = model.kneighbors(test_embeddings, n_neighbors=K)
                
                
                # Retrieve the neighbor classes using the indices.
                # This will result in an array of shape (n_test_samples, K)
                neighbor_classes = combined_labels[neighbor_indices]
                
                # Get the predicted classes for each test embedding (already computed)
                # y_pred is the predicted class for each test sample
                
                # Create column names for neighbor classes
                neighbor_cols = [f'neighbor{i+1}_class' for i in range(K)]
                
                # Create a DataFrame using the neighbor classes
                df_neighbors = pd.DataFrame(neighbor_classes, columns=neighbor_cols)
                
                # Create a DataFrame for the test indices and predicted classes
                df_test = pd.DataFrame({
                    'test_idx': test_indices,      # Assuming test_indices is already loaded from CSV
                    'predicted_class': y_pred,
                    'actual_class':test_labels
                })
                
                
                # Concatenate the test indices/predicted classes with the neighbor classes
                # Here, we assume that the order of rows in test_indices, y_pred, and neighbor_classes all match.
                df_result = pd.concat([df_test, df_neighbors], axis=1)
                
                # Optional: Rearrange columns to have test_idx, neighbor classes, then predicted_class.
                cols_order = ['test_idx'] + neighbor_cols + ['predicted_class'] + ['actual_class']
                df_result = df_result[cols_order]
                
                # Display the first few rows of the table
                print(df_result.head())
                
                # df_result.to_csv(f"../knn_results/test_neighbors_and_predictions_trial{t}.csv", index=False)
                
                # # Evaluate the model's performance
                accuracy = accuracy_score(test_labels, y_pred)
                print("Classification Accuracy:", accuracy)
                print("Classification F1 Score:", f1_score(test_labels, y_pred))
                
                print(confusion_matrix(test_labels, y_pred))
                
                # 4. Predict probabilities on the test set
                #    This returns an array of shape (n_samples, n_classes)
                y_pred_prob = model.predict_proba(test_embeddings)
                
                
                # For class 1 (the positive class), use the predicted probabilities for class 1
                auc_class1 = roc_auc_score(test_labels, y_pred_prob[:, 1])
                
                # For class 0, flip the labels and use the predicted probabilities for class 0.
                # Essentially, we treat "0" as the positive class.
                auc_class0 = roc_auc_score(1 - test_labels, y_pred_prob[:, 0])
                print("Test AUROC:",auc_class0)
                
                result_dict = {
                        'model': model,
                        'image_model': image_model,
                        'loss': 'supcon',
                        'input': input,
                        'trial': trial,
                        'test_auroc': auc_class0,
                        'F1 Score:':f1_score(test_labels, y_pred),

                    }
                print(result_dict)
                results.append(result_dict)
                
results_df = pd.DataFrame(results)
results_df.to_csv(f"../results/KNN_results_metatransformer_original_label.csv", index=False)
