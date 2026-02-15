#Multimodal Net



import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from .NeuMiss import NeuMissMLP 

class MultimodalNet(nn.Module):
    def __init__(self, args,output_dim=128):
        """
        Args:
            embed_dims (list): List of dimensions of embeddings for each modality.
            mlp_hidden_dims (list): List of hidden layer sizes for the MLP.
            output_dim (int): Dimension of the MLP output.
        """
        super(MultimodalNet, self).__init__()

        self.supcon_input=args.supcon_input.lower()

        
        self.image_projectionMLP=nn.Sequential(nn.Linear(512, 256),nn.ReLU(), nn.Linear(256,128))
        self.radreport_projection=nn.Sequential(nn.Linear(768, 256),nn.ReLU(), nn.Linear(256,128))
        self.analytes_encoder=NeuMissMLP(n_features=56, neumiss_depth=3, mlp_depth=3, output_dimension=128)
        self.vitals_encoder=NeuMissMLP(n_features=44, neumiss_depth=3, mlp_depth=3, output_dimension=128)
        self.demographics_encoder=NeuMissMLP(n_features=27, neumiss_depth=3, mlp_depth=3,  output_dimension=128)
        
        self.orders_encoder=NeuMissMLP(n_features=26, neumiss_depth=3, mlp_depth=3,  output_dimension=128)
        self.radreportlabel_encoder=NeuMissMLP(n_features=15, neumiss_depth=3, mlp_depth=3,  output_dimension=128)


     
        input_dim = 128*6# Concatenated embedding size
        self.mlp = nn.Sequential(nn.Linear(input_dim, int(input_dim/2)),
                                nn.ReLU(),
                                nn.Linear(int(input_dim/2),output_dim))

    def forward(self, images, analytes, vitals, orders, demographics, radreport,radreportlabels):
        """
        Args:
            embeddings (list of tensors): List of tensors for each modality (B, embed_dim).
        Returns:
            torch.Tensor: Output from the MLP (B, output_dim).
        """

        img=self.image_projectionMLP(images)
        ana=self.analytes_encoder(analytes)
        vit=self.vitals_encoder(vitals)
        demo=self.demographics_encoder(demographics)
        ord=self.orders_encoder(orders)
        
        if self.supcon_input=='label':
            radlabels=self.radreportlabel_encoder(radreportlabels)
            embeddings=[img, ana, vit, demo, ord,radlabels]
        else:
            #Rad Report
            radreports=self.radreport_projection(radreport)
            embeddings=[img, ana, vit, demo, ord,radreports]


        # Concatenate weighted embeddings
        concatenated_embedding = torch.cat(embeddings, dim=-1)  # (B, sum(embed_dims))

        # Pass through MLP
        output = self.mlp(concatenated_embedding)

  
        return F.normalize(output, p=2, dim=-1)

