import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from .NeuMiss import NeuMissMLP


class MetaTransformer(nn.Module):
    def __init__(self, args, embed_dim=128, depth=12, num_heads=8):
        """
        Args:
            args: experiment arguments (expects args.supcon_input)
            embed_dim (int): dimension of modality embeddings and transformer hidden dim
            depth (int): number of transformer blocks
            num_heads (int): number of attention heads
        """
        super(MetaTransformer, self).__init__()

        self.supcon_input = args.supcon_input.lower()

        # ---- Modality-specific encoders (all output [B, embed_dim]) ----
        self.image_projectionMLP = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, embed_dim)
        )
        self.radreport_projection = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, embed_dim)
        )
        self.analytes_encoder = NeuMissMLP(
            n_features=56, neumiss_depth=3, mlp_depth=3, output_dimension=embed_dim
        )
        self.vitals_encoder = NeuMissMLP(
            n_features=44, neumiss_depth=3, mlp_depth=3, output_dimension=embed_dim
        )
        self.demographics_encoder = NeuMissMLP(
            n_features=27, neumiss_depth=3, mlp_depth=3, output_dimension=embed_dim
        )
        self.orders_encoder = NeuMissMLP(
            n_features=26, neumiss_depth=3, mlp_depth=3, output_dimension=embed_dim
        )
        self.radreportlabel_encoder = NeuMissMLP(
            n_features=15, neumiss_depth=3, mlp_depth=3, output_dimension=embed_dim
        )

        # ---- Meta-transformer for fusion ----
        self.num_modalities = 6
        self.embed_dim = embed_dim

        # CLS + positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_modalities + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder (works directly at dim=128)
        self.encoder = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, images, analytes, vitals, orders, demographics,
                radreport, radreportlabels):
        """
        Returns:
            torch.Tensor: fused normalized embedding [B, embed_dim]
        """

        # ---- Encode each modality to [B, 128] ----
        img  = self.image_projectionMLP(images)
        ana  = self.analytes_encoder(analytes)
        vit  = self.vitals_encoder(vitals)
        demo = self.demographics_encoder(demographics)
        ords = self.orders_encoder(orders)

        if self.supcon_input == "label":
            rad = self.radreportlabel_encoder(radreportlabels)
        else:
            rad = self.radreport_projection(radreport)

        # Stack into [B, num_modalities, 128]
        embeddings = torch.stack([img, ana, vit, demo, ords, rad], dim=1)

        # ---- Transformer fusion ----
        B = embeddings.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)   # [B,1,128]
        x = torch.cat((cls_tokens, embeddings), dim=1)  # [B,7,128]

        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.encoder(x)                             # [B,7,128]
        x = self.norm(x)

        cls_out = x[:, 0]                               # [B,128]
        return F.normalize(cls_out, p=2, dim=-1)
