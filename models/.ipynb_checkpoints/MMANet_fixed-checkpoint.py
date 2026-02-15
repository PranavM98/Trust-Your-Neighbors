#Multimodal Attention



import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from .NeuMiss import NeuMissMLP
class MultimodalAttentionFixed(nn.Module):
    def __init__(self, args,output_dim=128):
        """
        Args:
            embed_dims (list): List of dimensions of embeddings for each modality.
            mlp_hidden_dims (list): List of hidden layer sizes for the MLP.
            output_dim (int): Dimension of the MLP output.
        """
        super(MultimodalAttentionFixed, self).__init__()

        self.supcon_input=args.supcon_input.lower()

        
        self.image_projectionMLP=nn.Sequential(nn.Linear(512, 256),nn.ReLU(), nn.Linear(256,128))
        self.radreport_projection=nn.Sequential(nn.Linear(768, 256),nn.ReLU(), nn.Linear(256,128))
        self.analytes_encoder=NeuMissMLP(n_features=56, neumiss_depth=3, mlp_depth=3, output_dimension=128)
        self.vitals_encoder=NeuMissMLP(n_features=44, neumiss_depth=3, mlp_depth=3, output_dimension=128)
        self.demographics_encoder=NeuMissMLP(n_features=27, neumiss_depth=3, mlp_depth=3,  output_dimension=128)
        
        self.orders_encoder=NeuMissMLP(n_features=26, neumiss_depth=3, mlp_depth=3,  output_dimension=128)
        self.radreportlabel_encoder=NeuMissMLP(n_features=15, neumiss_depth=3, mlp_depth=3,  output_dimension=128)



        #adding radiology reports

        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(i, 64),  # Project to intermediate dim
                nn.ReLU(),
                nn.Linear(64, 1)          # Attention score
            )
            for i in [128,128,128,128,128,128]
        ])


        # MLP to process concatenated embedding

        input_dim = 128+128+128+128+128+128# Concatenated embedding size
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

        attention_scores = []
        weighted_embeddings = []

        # Compute attention scores for all modalities
        raw_scores = torch.cat([self.attention_layers[i](emb) for i, emb in enumerate(embeddings)], dim=1)  # (B, M)

        # # Apply softmax across the modality dimension (dim=1)
        # attention_weights = F.softmax(raw_scores, dim=1)  # Normalize across modalities (M)
        # print(attention_weights.shape)
        attention_weights=[0.21564874343318877, 0.0779552192131047, 0.03976965079351809, 0.15241153682597108, 0.22761580951543717, 0.2865990402187803]
        # Convert to a tensor
        attention_weights_tensor = torch.tensor(attention_weights)

        # Repeat 150 times along a new dimension
        attention_weights_repeated = attention_weights_tensor.unsqueeze(0).repeat(images.shape[0], 1)

        # Weight embeddings by the computed attention scores
        for i, emb in enumerate(embeddings):
            weighted_embeddings.append(emb * attention_weights_repeated[:, i].unsqueeze(-1).to('cuda:0'))  # (B, D)


        # # Compute attention scores and weight embeddings
        # for i, emb in enumerate(embeddings):
        #     score = self.attention_layers[i](emb)  # (B, 1)
        #     score = F.softmax(score, dim=0)       # Normalize across modalities
        #     attention_scores.append(score)
        #     weighted_embeddings.append(emb * score)  # Weight embedding by score

        # Concatenate weighted embeddings
        concatenated_embedding = torch.cat(weighted_embeddings, dim=-1)  # (B, sum(embed_dims))

        # Pass through MLP
        output = self.mlp(concatenated_embedding)

  
        return F.normalize(output, p=2, dim=-1), attention_weights

