"""
Multimodal Model Training Script
--------------------------------
This script handles end-to-end training, validation, and testing of multimodal models
(e.g., MMNet, MMANet, MetaTransformer) using SupCon or CE losses. It supports saving
embeddings and models at the best validation epoch, along with TensorBoard logging.

Usage Example:
--------------
python model_training.py \
    --modality tabular \
    --epoch 500 \
    --model mmanet \
    --device cuda \
    --gpu 1
"""

import os
import sys
import time
import datetime
import argparse
import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Third-party libraries
from pytorch_metric_learning import miners
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from torchsummary import summary
from torchvision import transforms
from torch_geometric.utils import dense_to_sparse

# MONAI utilities
from monai.transforms import Compose, LoadImage, Resize, ScaleIntensity
from monai.config import DtypeLike

# Local imports
from final_utils import data_preprocessing
from run_params import RunParameters
from models.MMNet import MultimodalNet
from models.MMANet import MultimodalAttention
from models.MMANet_fixed import MultimodalAttentionFixed
from models.MetaTransformer import MetaTransformer
from models.MMBANet import MultimodalBiAttention
from losses.supcon import SupConLoss


# -----------------------------
# Argument Parser Configuration
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='mmanet',
                    help='Model: mmanet | mmnet | mmbanet | metatransformer')
parser.add_argument('--image_model', type=str, default='ctnet',
                    help='Image encoder: 3dresnet | position_model | ctclip')
parser.add_argument('--supcon_input', type=str, default='report',
                    help='Input type for SupCon: label | report')
parser.add_argument('--trial', type=int, default=100, help='Trial number')
parser.add_argument('--epoch', type=int, default=150, help='Number of training epochs')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--temperature', type=float, default=0.1, help='SupCon temperature')
parser.add_argument('--momentum', type=float, default=0.9, help='Optimizer momentum')
parser.add_argument('--device', type=str, default="cuda", help='Device: cuda | cpu')
parser.add_argument('--gpu', type=int, default=0, help='GPU index (0, 1, 2, ...)')
parser.add_argument('--loss', type=str, default='supcon',
                    help='Loss function: weighted_supcon | supcon | ce')
parser.add_argument('--data_text_split', type=str, default='original',
                    help='Data split type: original | temporal | institutional')
parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer: sgd | adam')
parser.add_argument('--train_batch_size', type=int, default=150, help='Train batch size')
parser.add_argument('--val_batch_size', type=int, default=150, help='Validation batch size')
parser.add_argument('--test_batch_size', type=int, default=500, help='Test batch size')

args = parser.parse_args()


# -----------------------------
# Device Configuration
# -----------------------------
if args.device == 'cuda':
    device_str = f"cuda:{args.gpu}"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using CUDA device: {device_str}")
else:
    device = torch.device("cpu")
    print("Using CPU")

torch.cuda.empty_cache()
print("Device:", device)


# -----------------------------
# TensorBoard Initialization
# -----------------------------
program_date_time = datetime.datetime.now()
folder = (
    f"final_runs/runs/{args.model.lower()}/"
    f"{args.image_model.lower()}_{args.loss.lower()}_"
    f"{args.data_text_split.lower()}_{args.supcon_input}_{args.trial}/"
)
writer = SummaryWriter(folder)


# -----------------------------
# Model Evaluation Functions
# -----------------------------
def model_testing(model, test_dataloader, saving_folder_epoch, name):
    """
    Evaluate model on test data and save embeddings.

    Args:
        model (torch.nn.Module): Trained model.
        test_dataloader (DataLoader): Test dataset loader.
        saving_folder_epoch (str): Folder name for saving embeddings.
        name (str): File prefix (e.g., 'testing').

    Returns:
        pd.DataFrame: DataFrame containing embeddings, labels, and indices.
    """
    global device, args
    model.eval()
    emb_list, target_list, idx_list = [], [], []

    with torch.no_grad():
        for data in tqdm.tqdm(test_dataloader, desc="Testing"):
            images, analytes, vitals, orders, demographics, radreport, radreportlabels, labels, idx = (
                data['img'].to(device).float(),
                data['analytes'].to(device).float(),
                data['vitals'].to(device).float(),
                data['orders'].to(device).int(),
                data['demographics'].to(device).float(),
                data['radreport'].to(device).float(),
                data['radreportlabels'].to(device).int(),
                data['outcome'],
                list(data['idx']),
            )

            if args.model.lower() in ['mmnet', 'metatransformer']:
                normalized_output = model(images, analytes, vitals, orders, demographics,
                                          radreport, radreportlabels)
            else:
                normalized_output, _ = model(images, analytes, vitals, orders, demographics,
                                             radreport, radreportlabels)

            emb_list.append(normalized_output.detach().cpu().numpy())
            target_list.append(labels.detach().cpu().numpy().flatten())
            idx_list.extend(idx)

    emb_array = np.concatenate(emb_list, axis=0)
    target_array = np.concatenate(target_list, axis=0)

    df = pd.DataFrame(emb_array)
    df['label'] = target_array
    df['idx'] = idx_list

    save_dir = (
        f"final_runs/runs/{args.model.lower()}/"
        f"{args.image_model.lower()}_{args.loss.lower()}_"
        f"{args.data_text_split.lower()}_{args.supcon_input}_{args.trial}/Epoch_{saving_folder_epoch}/"
    )
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, f"{name}_emb.csv"), index=False)

    return df


def model_validation(model, optimizer, criterion, val_dataloader, epoch):
    """
    Run validation loop to compute average loss and embeddings.

    Args:
        model (torch.nn.Module): Model being validated.
        optimizer: Optimizer used for training.
        criterion: Loss function.
        val_dataloader (DataLoader): Validation data loader.
        epoch (int): Current epoch number.

    Returns:
        tuple: (validation loss, DataFrame of embeddings and labels)
    """
    global device, args
    model.eval()
    val_running_loss = 0.0
    emb_list, target_list, idx_list = [], [], []

    with torch.no_grad():
        for data in tqdm.tqdm(val_dataloader, desc='Validation', position=2, leave=False):
            images, analytes, vitals, orders, demographics, radreport, radreportlabels, labels, idx = (
                data['img'].to(device).float(),
                data['analytes'].to(device).float(),
                data['vitals'].to(device).float(),
                data['orders'].to(device).int(),
                data['demographics'].to(device).float(),
                data['radreport'].to(device).float(),
                data['radreportlabels'].to(device).int(),
                data['outcome'],
                list(data['idx']),
            )

            if args.model.lower() in ['mmnet', 'metatransformer']:
                outputs = model(images, analytes, vitals, orders, demographics, radreport, radreportlabels)
            else:
                outputs, _ = model(images, analytes, vitals, orders, demographics, radreport, radreportlabels)

            loss = criterion(outputs, labels)
            emb_list.append(outputs.detach().cpu().numpy())
            target_list.append(labels.detach().cpu().numpy().flatten())
            idx_list.extend(idx)
            val_running_loss += loss.item() * images.size(0)

    emb_array = np.concatenate(emb_list, axis=0)
    target_array = np.concatenate(target_list, axis=0)
    df = pd.DataFrame(emb_array)
    df['label'] = target_array
    df['idx'] = idx_list

    return val_running_loss, df


def model_training(model, optimizer, criterion, train_dataloader, val_dataloader):
    """
    Train model across epochs with validation and checkpointing. 
    The epoch with the best val loss is stored with the train, val embeddings as well as the model weights.

    Args:
        model (torch.nn.Module): Model to train.
        optimizer: Optimizer instance.
        criterion: Loss function.
        train_dataloader (DataLoader): Training data loader.
        val_dataloader (DataLoader): Validation data loader.

    Returns:
        torch.nn.Module: Trained model.
    """
    global device, args, writer
    num_epochs = args.epoch
    best_val_loss = float('inf')

    for epoch in tqdm.tqdm(range(num_epochs), desc='Training'):
        model.train()
        running_loss = 0.0

        for data in tqdm.tqdm(train_dataloader, desc='Train Batches', position=1, leave=False):
            images, analytes, vitals, orders, demographics, radreport, radreportlabels, labels, idx = (
                data['img'].to(device).float(),
                data['analytes'].to(device).float(),
                data['vitals'].to(device).float(),
                data['orders'].to(device).int(),
                data['demographics'].to(device).float(),
                data['radreport'].to(device).float(),
                data['radreportlabels'].to(device).int(),
                data['outcome'],
                list(data['idx']),
            )

            optimizer.zero_grad()
            if args.model.lower() in ['mmnet', 'metatransformer']:
                outputs = model(images, analytes, vitals, orders, demographics, radreport, radreportlabels)
            else:
                outputs, _ = model(images, analytes, vitals, orders, demographics, radreport, radreportlabels)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        avg_train_loss = running_loss / len(train_dataloader.dataset)
        writer.add_scalar('training_loss', avg_train_loss, epoch)

        # Validation phase
        val_loss, val_df = model_validation(model, optimizer, criterion, val_dataloader, epoch)
        avg_val_loss = val_loss / len(val_dataloader.dataset)
        writer.add_scalar('validation_loss', avg_val_loss, epoch)

        # Save best model checkpoint
        if avg_val_loss < best_val_loss or epoch == 0:
            best_val_loss = avg_val_loss
            epoch_folder = os.path.join(folder, f"Epoch_{epoch}")
            os.makedirs(epoch_folder, exist_ok=True)

            torch.save(model.state_dict(), os.path.join(epoch_folder, f"model_{epoch}.pth"))

            # Save embeddings
            emb_list, target_list, idx_list = [], [], []
            model.eval()
            with torch.no_grad():
                for data in train_dataloader:
                    images, analytes, vitals, orders, demographics, radreport, radreportlabels, labels, idx = (
                        data['img'].to(device).float(),
                        data['analytes'].to(device).float(),
                        data['vitals'].to(device).float(),
                        data['orders'].to(device).int(),
                        data['demographics'].to(device).float(),
                        data['radreport'].to(device).float(),
                        data['radreportlabels'].to(device).int(),
                        data['outcome'],
                        list(data['idx']),
                    )
                    outputs = (
                        model(images, analytes, vitals, orders, demographics, radreport, radreportlabels)
                        if args.model.lower() in ['mmnet', 'metatransformer']
                        else model(images, analytes, vitals, orders, demographics, radreport, radreportlabels)[0]
                    )
                    emb_list.append(outputs.detach().cpu().numpy())
                    target_list.append(labels.detach().cpu().numpy().flatten())
                    idx_list.extend(idx)

            emb_array = np.concatenate(emb_list, axis=0)
            target_array = np.concatenate(target_list, axis=0)
            df = pd.DataFrame(emb_array)
            df['label'] = target_array
            df['idx'] = idx_list
            df.to_csv(os.path.join(epoch_folder, "training_emb.csv"), index=False)
            val_df.to_csv(os.path.join(epoch_folder, "validation_emb.csv"), index=False)

            # Test phase
            model_testing(model.to(device), test, str(epoch), 'testing')

    return model


def model_definition(run_params):
    """
    Define the model, optimizer, and criterion based on args.

    Args:
        run_params (RunParameters): Run configuration tracker.

    Returns:
        tuple: (model, optimizer, criterion)
    """
    global device, args

    if args.model.lower() == 'mmnet':
        model = MultimodalNet(args)
    elif args.model.lower() == 'mmanet':
        model = MultimodalAttention(args)
    elif args.model.lower() == 'metatransformer':
        model = MetaTransformer(args)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    print("GPU available:", torch.cuda.is_available())

    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    criterion = SupConLoss(args)
    print("Criterion:", criterion)

    # Update run parameters
    run_params.update_model_params('learning_rate', args.learning_rate)
    run_params.update_model_params('momentum', args.momentum)
    run_params.update_model_params('optimizer', args.optimizer.lower())
    run_params.update_model_params('loss', args.loss.lower())
    run_params.update_model_params('loss_input', args.supcon_input.lower())
    run_params.update_model_params('temperature_supcon', args.temperature)
    run_params.update_model_params('epoch', args.epoch)

    return model.to(device), optimizer, criterion.to(device)


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == '__main__':
    folder = (
        f"final_runs/runs/{args.model.lower()}/"
        f"{args.image_model.lower()}_{args.loss.lower()}_"
        f"{args.data_text_split.lower()}_{args.supcon_input}_{args.trial}/"
    )

    parameters = RunParameters(folder=folder)
    train, val, test = data_preprocessing(args, parameters)
    print("Training, Validation, Testing split completed.")

    model, optimizer, criterion = model_definition(parameters)
    parameters.save_to_yaml()

    model = model_training(model=model, optimizer=optimizer, criterion=criterion,
                           train_dataloader=train, val_dataloader=val)

    writer.close()
