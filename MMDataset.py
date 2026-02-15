import torch
from torch.utils.data import Dataset
from monai.transforms import Compose, LoadImage, Resize, ScaleIntensity
from monai.config import DtypeLike
import pandas as pd
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import torch.nn.functional as F
from itertools import chain




def generate_column_list(filename):

    with open(filename, "r") as file:
        column_list=[]
        for line in file:
            # Strip whitespace and check if the line is not empty and doesn't start with '#'
            stripped_line = line.strip()
            if stripped_line and not stripped_line.startswith("#"):
                column_list.append(stripped_line)
    return column_list

class MMDataset(Dataset):
    def __init__(self, data):

        """
        Args:
            data (pd.DataFrame): The dataframe containing your tabular data.
            target_column (str): The column name of the target variable.
        """

        analytes_columns=generate_column_list("important_data/analytes_important_columns.txt")
        vitals_columns=generate_column_list("important_data/vitals_important_columns.txt")
        demographics_columns=generate_column_list("important_data/demographics_important_columns.txt")
        orders_columns=generate_column_list("important_data/patorders_important_columns.txt")
        
        radreportlabels_columns=generate_column_list("important_data/radreportlabels_important_columns.txt")

        radreportemb_columns=[col for col in data.columns if col.startswith("text_")]

        imgemb_columns=[col for col in data.columns if col.startswith("img_")]


        self.imgemb=data[imgemb_columns].values
        self.textemb=data[radreportemb_columns].values

        #Tabular
        self.analytesemb=data[analytes_columns].values
        self.vitalsemb=data[vitals_columns].values
        self.ordersemb=data[orders_columns].values
        self.demographicsemb=data[demographics_columns].values
        
        
        self.rrlabelemb=data[radreportlabels_columns].values
        self.outcomes=data['label'].values
        self.accnums=data['ACC_NUM'].values
      

    def __len__(self):
        return len(self.outcomes)

    def __getitem__(self, idx):

        data={'img':self.imgemb[idx],
        'analytes':self.analytesemb[idx],
        'vitals':self.vitalsemb[idx],
        'orders':self.ordersemb[idx],
        'demographics':self.demographicsemb[idx],
        'radreport':self.textemb[idx],
        'radreportlabels':self.rrlabelemb[idx],
        'outcome':self.outcomes[idx],
        'idx':self.accnums[idx]
        }
        return data