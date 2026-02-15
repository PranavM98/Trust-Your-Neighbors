import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import rac_utils
from models.graphsage import GraphSAGE
from models.gat import GAT
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
import numpy as np
import pandas as pd
import os
import re
import pandas as pd
import argparse
import os
import utils
import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



# Training loop
def train(model, data, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    # print(out)
    # print(data.train_mask)
    loss = criterion(out[0][data.train_mask], torch.tensor(data.y[data.train_mask], dtype=torch.float32).unsqueeze(1))
    loss.backward()
    optimizer.step()
    return loss.item(), model


import torch
from sklearn.metrics import roc_auc_score

def evaluate(mask, data, model):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)  # Model prediction

        # Apply sigmoid to get probabilities for the positive class
        probs = torch.sigmoid(out[0]).squeeze()  # shape: [num_nodes]

        # Select only the nodes specified by the mask
        probs_masked = probs[mask]
        true_labels = data.y[mask]  # should be binary (0 or 1)

        # Move data to CPU and detach from the computation graph for use with scikit-learn
        probs_masked_np = probs_masked.detach().cpu().numpy()
        true_labels_np = true_labels.detach().cpu().numpy()

        # Compute AUROC for class 1 (positive class) directly
        auroc_class1 = roc_auc_score(true_labels_np, probs_masked_np)

        # Compute AUROC for class 0 (negative class) using inverted probabilities
        auroc_class0 = roc_auc_score(1 - true_labels_np, 1 - probs_masked_np)

        # Average the two AUROCs
        average_auroc = (auroc_class0 + auroc_class1) / 2

        print("AUROC for Class 0:", auroc_class0)
        print("AUROC for Class 1:", auroc_class1)
        print("Average AUROC:", average_auroc)

        return average_auroc, auroc_class0, auroc_class1





def graph_main(folder):
    data,x,y=rac_utils.graph_preprocessing(folder)

    # Check for GPU
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    #device=torch.device("cpu")
    print(f"Using device: {device}")
    data = data.to(device)

    x, y = x.to(device), y.to(device)
    model_name='GraphSAGE'

    print("MODEL NAME: ", model_name)
    if model_name=='GraphSAGE':
        print("Initialized GraphSAGE")
        print(x.shape)

        # Initialize model, optimizer, and loss function
        model = GraphSAGE(in_channels=x.shape[1], hidden_channels=128, out_channels=1).to(device)  # Adjust hidden size/out_channels as needed

    elif model_name=='GAT':
        print("Initialized GAT")
        # Replace GraphSAGE with GAT
        model = GAT(
            in_channels=x.shape[1], 
            hidden_channels=128, 
            out_channels=1, 
            heads=8,  # Number of attention heads
            dropout=0.3  # Dropout probability
        ).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.BCEWithLogitsLoss()


    # Main training process
    best_val_auroc = 0
    best_epoch = 0

    for epoch in range(200):
        loss, model = train(model,data, criterion, optimizer)
        train_auroc, train_auc_class_0, auc_class_1 = evaluate(data.train_mask,data,model)
        val_auroc, val_auc_class_0, val_auc_class_1 = evaluate(data.val_mask,data,model)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Train AUROC: {train_auroc:.4f}, Val AUROC: {val_auroc:.4f}")
        
        if val_auroc > best_val_auroc:
            best_epoch = epoch
            best_val_auroc = val_auroc

    # Retrain on train + val
    combined_train_mask = data.train_mask | data.val_mask
    print("Best Epoch: ", best_epoch)
    print("-----------------Retraining on Train and Val-----------------")


    print("Initialized GraphSAGE for Combined")
    # Initialize model, optimizer, and loss function
    model = GraphSAGE(in_channels=x.shape[1], hidden_channels=128, out_channels=1).to(device)  # Adjust hidden size/out_channels as needed
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)



    for epoch in range(best_epoch):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[0][combined_train_mask], torch.tensor(data.y[combined_train_mask], dtype=torch.float32).unsqueeze(1)) 
        loss.backward()
        optimizer.step()

    # Test evaluation
    print("--------TEST--------------")
    test_auroc, auroc_class_0, auroc_class_1= evaluate(data.test_mask, data, model)
    print(f"Test AUC: {test_auroc:.4f}")
    return test_auroc, auroc_class_0, auroc_class_1






parser = argparse.ArgumentParser()
parser.add_argument('--trial', type=str, default='1', help='1,2, 3')
args = parser.parse_args()

# given image_model loss loss _input
folder='/storage/pm231-projects/Pro00103826 - TBI predictive model/Pranav Manjunath/final_MICCAI2025/new_experiments_rebuttal/final_runs/runs/'


results = []
for model in ['metatransformer']:
    for image_model in ['ctclip','ctnet','3dresnet']:
        for input in ['label']:
            torch.cuda.empty_cache()
            for t in ['100']:
                trial=t

                file_path=folder+model+'/'+image_model+"_supcon_original_label_"+f"{trial}"
                final_folder=utils.get_last_epoch_data(file_path)
                test_auroc, auroc_0, auroc_1 =graph_main(final_folder)

                        
                result_dict = {
                        'model': model,
                        'image_model': image_model,
                        'loss': 'supcon',
                        'input': input,
                        'trial': trial,
                        'test_auroc': test_auroc,
                        'class_0_auroc':auroc_0,
                        'class_1_auroc':auroc_1,

                    }
                print(result_dict)
                results.append(result_dict)
                torch.cuda.empty_cache()


results_df = pd.DataFrame(results)
results_df.to_csv(f"results/meta_transformer_graphSAGE_label.csv", index=False)









# folder='/storage/pm231-projects/Pro00103826 - TBI predictive model/Pranav Manjunath/final_MICCAI2025/final_runs/runs'




# results = []
# for model in ['mmnet','mmbanet','mmanet']:
#     for image_model in ['ctnet']:
#             for input in ['label','report']:
#                 for trial in ['2']:
#                     file_prefix=f'{image_model}_supcon_{input}_{trial}'
#                     final_folder=folder+'/'+f'{model}'+'/'+file_prefix
#                     torch.cuda.empty_cache()
#                     for epoch in ['60','61','62','63']:
                    
#                         try:
#                             file_path=final_folder+f'/Epoch_{epoch}/'
#                             print(file_path)
#                             test_auroc, auroc_0, auroc_1 =graph_main(file_path)
#                             break
#                         except:
#                             continue

                    
                    
  
#                 result_dict = {
#                     'model': model,
#                     'image_model': image_model,
#                     'loss': 'supcon',
#                     'input': input,
#                     'trial': trial,
#                     'test_auroc': test_auroc,
#                     'class_0_auroc':auroc_0,
#                     'class_1_auroc':auroc_1,

#                 }
#                 print(result_dict)
#                 results.append(result_dict)
                    
#                     #3dresnet_supcon_label_2_2025-01-24 14:49:02.873829
