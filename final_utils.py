"""
Data Preprocessing and Merging Pipeline for Multimodal TBI Dataset
-----------------------------------------------------------------
This module prepares multimodal data (imaging, labs, vitals, demographics, orders,
and reports) for downstream model training.

It handles:
- Cohort subsetting and filtering
- Temporal, institutional, or original train/val/test splits
- Merging and alignment of multimodal features
- Missing data checks and CSV export
- Conversion to PyTorch Datasets and DataLoaders

Author: Pranav Manjunath
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    roc_auc_score,
)
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from monai.transforms import Compose, LoadImage, Resize, ScaleIntensity
from monai.config import DtypeLike
from transformers import BertModel, BertConfig, BertTokenizer
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier

from MMDataset import MMDataset

# Global seed
random_state = 42


# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------
def subset_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out invalid cases and subset the dataset for training/testing.

    Steps:
    - Drop rows with missing radiology reports.
    - Remove reports that reference non-head CT procedures.
    - Perform an 80/20 random train/test split for downstream stratification.

    Args:
        data (pd.DataFrame): Full dataset.

    Returns:
        pd.DataFrame: Subset of valid data for downstream splitting.
    """
    global random_state

    # Drop rows with missing reports
    data = data.dropna(subset=["narrative"])

    # Remove cases with irrelevant CT reports (e.g., cervical spine)
    procedures = ["cta", "cervical spine", "ct cervical"]
    pat_enc_remove = []

    for i, row in data.iterrows():
        report = row["narrative"]
        if any(proc in report.lower() for proc in procedures):
            pat_enc_remove.append(row["pat_enc_csn_id"])

    data = data[~data["pat_enc_csn_id"].isin(pat_enc_remove)]

    # Downsample if needed
    train_df, _ = train_test_split(data, test_size=0.2, random_state=random_state)
    return train_df


def extracting_acc_nums(data: pd.DataFrame, text_split: str):
    """
    Generate train/validation/test ACC_NUM splits.

    Supports three modes:
        - 'original': Random stratified split
        - 'temporal': Year-based chronological split
        - 'institutional': Hospital-based split

    Args:
        data (pd.DataFrame): Raw cohort dataframe.
        text_split (str): Split type ('original', 'temporal', or 'institutional').

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Train, validation, and test ACC_NUM arrays.
    """
    global random_state
    data = subset_data(data)
    cohort = pd.read_csv(
            "Cohort File"
    )

    if text_split == "original":
        print("Original Split")
        train, test = train_test_split(
            data, stratify=data["label"], test_size=0.25, random_state=random_state
        )
        train, val = train_test_split(
            train, stratify=train["label"], test_size=0.25, random_state=random_state
        )
        train_acc, val_acc, test_acc = (
            train["ACC_NUM"].values,
            val["ACC_NUM"].values,
            test["ACC_NUM"].values,
        )

    elif text_split == "temporal":
        print("Temporal Split")
        mapping = cohort[["emergency_admit_time", "ACC_NUM"]]
        combined = data.merge(mapping, on="ACC_NUM", how="left")
        combined["emergency_admit_time_y"] = pd.to_datetime(
            combined["emergency_admit_time_y"]
        ).dt.year

        train = combined[combined["emergency_admit_time_y"] < 2018]
        val = combined[combined["emergency_admit_time_y"] == 2018]
        test = combined[combined["emergency_admit_time_y"] == 2019]

        train_acc, val_acc, test_acc = (
            train["ACC_NUM"].values,
            val["ACC_NUM"].values,
            test["ACC_NUM"].values,
        )

        print(f"Train: {len(train_acc)} | Val: {len(val_acc)} | Test: {len(test_acc)}")

    elif text_split == "institutional":
        print("Institutional Split")
        mapping = cohort[["location", "ACC_NUM"]]
        combined = data.merge(mapping, on="ACC_NUM", how="left")

        training = combined[combined["location_y"] == "DUKE RALEIGH HOSPITAL"]
        train, val = train_test_split(
            training, stratify=training["label"], test_size=0.20
        )
        test = combined[combined["location_y"] == "DUKE UNIVERSITY HOSPITAL"]

        train_acc, val_acc, test_acc = (
            train["ACC_NUM"].values,
            val["ACC_NUM"].values,
            test["ACC_NUM"].values,
        )

        print(f"Train: {len(train_acc)} | Val: {len(val_acc)} | Test: {len(test_acc)}")

    else:
        raise ValueError(f"Unknown text split: {text_split}")

    return train_acc, val_acc, test_acc


def add_prefix(data: pd.DataFrame, data_type: str = "text") -> list[str]:
    """
    Prefix column names with a modality tag (e.g., text_ or img_).

    Args:
        data (pd.DataFrame): DataFrame with embedding features.
        data_type (str): 'text' or 'img'.

    Returns:
        list[str]: Prefixed column names.
    """
    prefix = "text_" if data_type == "text" else "img_"
    return [f"{prefix}{col}" if col.isdigit() else col for col in data.columns]


def missing_value_test(data: pd.DataFrame, prefix: str = "img_") -> int:
    """
    Check for missing values in columns with a given prefix.

    Args:
        data (pd.DataFrame): Input DataFrame.
        prefix (str): Prefix (e.g., 'img_', 'text_').

    Returns:
        int: Count of columns with missing values.
    """
    columns = [col for col in data.columns if col.startswith(prefix)]
    missing_values = data[columns].isnull().sum()
    return len(missing_values[missing_values > 0])


def merge_data(acc_num_list, args):
    """
    Merge all modality-specific dataframes using ACC_NUM as the key.

    Args:
        acc_num_list (list): List of accession numbers to include.
        args: Command-line arguments containing split and model info.

    Returns:
        pd.DataFrame: Fully merged multimodal dataframe.
    """
    # Load modality-specific data
    base_path = (
        "/storage/pm231-projects/Pro00103826 - TBI predictive model/"
        "data_pipeline_v1/Results/folder_ER_data/MICCAI2025/"
    )
    analytes = pd.read_csv(os.path.join(base_path, "analytes_dataframe.csv"))
    vitals = pd.read_csv(os.path.join(base_path, "vitals_dataframe.csv"))
    outcome = pd.read_csv(os.path.join(base_path, "outcome_dataframe.csv"))
    demographics = pd.read_csv(os.path.join(base_path, "patdemographics_dataframe.csv"))
    orders = pd.read_csv(os.path.join(base_path, "patorders_dataframe.csv"))
    radreportlabels = pd.read_csv(os.path.join(base_path, "radreportlabels_dataframe.csv"))
    radreportemb = pd.read_csv(os.path.join(base_path, "radreportemb_dataframe.csv"))

    # Map sex labels and encode reasons
    demographics["sex_label"] = demographics["sex_label"].map({"Male": 0, "Female": 1})
    reason_cols = [col for col in demographics.columns if col.startswith("erreason_")]
    demographics[reason_cols] = demographics[reason_cols].astype(int)

    # Image embeddings by model type
    img_path = os.path.join(base_path, f"imgemb_{args.image_model.lower()}_dataframe.csv")
    imgemb = pd.read_csv(img_path)

    # Extract ACC_NUM from filename
    imgemb["ACC_NUM"] = imgemb["ACC_NUM"].apply(
        lambda x: os.path.splitext(os.path.basename(x))[0]
    )

    # Add modality prefixes
    radreportemb.columns = add_prefix(radreportemb, data_type="text")
    imgemb.columns = add_prefix(imgemb, data_type="img")

    # Merge all modalities
    merged_df = (
        outcome.merge(analytes, on="ACC_NUM", how="outer")
        .merge(vitals, on="ACC_NUM", how="outer")
        .merge(demographics, on="ACC_NUM", how="outer")
        .merge(orders, on="ACC_NUM", how="outer")
        .merge(radreportlabels, on="ACC_NUM", how="outer")
        .merge(radreportemb, on="ACC_NUM", how="outer")
        .merge(imgemb, on="ACC_NUM", how="outer")
    )

    merged_df = merged_df[merged_df["ACC_NUM"].isin(acc_num_list)]
    merged_df["label"] = merged_df["label"].apply(lambda x: 1 if x in [1, 2, 3] else 0)

    # Missing data checks
    img_missing = missing_value_test(merged_df, prefix="img_")
    text_missing = missing_value_test(merged_df, prefix="text_")

    if img_missing == 0 and text_missing == 0:
        merged_df.to_csv(f"Dataset/merged_dataset_{args.data_text_split}.csv", index=False)
        print("Image and text embeddings complete — dataset saved.")
        return merged_df
    else:
        print("Missing data detected — exiting.")
        exit()


def data_update_params(run_param, type_data, y_data):
    """
    Update dataset metadata in RunParameters with label distribution.

    Args:
        run_param: RunParameters object.
        type_data (str): 'train', 'val', or 'test'.
        y_data (np.ndarray): Labels array.
    """
    run_param.update_data_params("size", len(y_data), parent_key=type_data)
    value_counts = pd.Series(y_data).value_counts()
    for idx, label in enumerate(value_counts.index):
        run_param.update_data_params(
            f"Class_{label}", int(value_counts[label]), parent_key=type_data
        )


def categorize_gcs(series: pd.Series) -> np.ndarray:
    """
    Categorize GCS numeric scores into clinical severity bins.

    Args:
        series (pd.Series): GCS score values.

    Returns:
        np.ndarray: Array of ['Mild', 'Moderate', 'Severe'] categories.
    """
    series = series.dropna()
    rounded = series.round()

    conditions = [
        (rounded >= 13) & (rounded <= 15),
        (rounded >= 9) & (rounded <= 12),
        (rounded >= 3) & (rounded <= 8),
    ]
    choices = ["Mild", "Moderate", "Severe"]

    return np.select(conditions, choices, default=np.nan)


def data_preprocessing(args, run_params):
    """
    Main data preprocessing function.

    Loads and merges multimodal features, performs cohort splitting,
    applies GCS categorization, and returns PyTorch DataLoaders.

    Args:
        args: Command-line arguments.
        run_params: RunParameters object for tracking metadata.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]:
            (train_dataloader, val_dataloader, test_dataloader)
    """
    global random_state

    data_file_name = "...Path To Dataset..."
    data = pd.read_csv(data_file_name)

    train_acc, val_acc, test_acc = extracting_acc_nums(data, args.data_text_split)
    train_df = merge_data(train_acc, args)
    val_df = merge_data(val_acc, args)
    test_df = merge_data(test_acc, args)

    # GCS binning
    for name, df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
        df = df.dropna(subset=["gcs_mean"]).copy()
        df["gcs_bin"] = categorize_gcs(df["gcs_mean"])
        print(f"{name} GCS distribution:\n{df['gcs_bin'].value_counts()}\n")

    # Metadata updates
    for split, labels in zip(
        ["train", "val", "test"],
        [train_df["label"].values, val_df["label"].values, test_df["label"].values],
    ):
        data_update_params(run_params, split, labels)

    # Convert to PyTorch datasets
    train, val, test = MMDataset(train_df), MMDataset(val_df), MMDataset(test_df)

    train_dataloader = DataLoader(train, batch_size=args.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=args.val_batch_size, shuffle=False)
    test_dataloader = DataLoader(test, batch_size=args.test_batch_size, shuffle=False)

    # Save batch sizes
    run_params.update_data_params("batch_size", args.train_batch_size, parent_key="train")
    run_params.update_data_params("batch_size", args.val_batch_size, parent_key="val")
    run_params.update_data_params("batch_size", args.test_batch_size, parent_key="test")

    return train_dataloader, val_dataloader, test_dataloader
