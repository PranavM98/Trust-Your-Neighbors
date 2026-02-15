import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from .NeuMiss import NeuMissMLP

class MetaTransformer(nn.Module):
    def __init__(self, args, output_dim=128, hidden_dim=768, depth=12, num_heads=12):
        """
        Args:
            args: experiment arguments (expects args.supcon_input)
            output_dim (int): final fused embedding size
            hidden_dim (int): transformer hidden dim (default 768 like ViT-base)
            depth (int): number of transformer blocks
            num_heads (int): number of attention heads
        """
        super(MetaTransformer, self).__init__()

        self.supcon_input = args.supcon_input.lower()

        # ---- Modality-specific encoders ----
        self.image_projectionMLP = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128)
        )
        self.radreport_projection = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 128)
        )
        self.analytes_encoder = NeuMissMLP(
            n_features=56, neumiss_depth=3, mlp_depth=3, output_dimension=128
        )
        self.vitals_encoder = NeuMissMLP(
            n_features=44, neumiss_depth=3, mlp_depth=3, output_dimension=128
        )
        self.demographics_encoder = NeuMissMLP(
            n_features=27, neumiss_depth=3, mlp_depth=3, output_dimension=128
        )
        self.orders_encoder = NeuMissMLP(
            n_features=26, neumiss_depth=3, mlp_depth=3, output_dimension=128
        )
        self.radreportlabel_encoder = NeuMissMLP(
            n_features=15, neumiss_depth=3, mlp_depth=3, output_dimension=128
        )

        # ---- Meta-transformer for fusion ----
        self.num_modalities = 6
        self.input_proj = nn.Linear(128, hidden_dim)  # project to transformer space

        # CLS + positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_modalities + 1, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder (your Block logic)
        self.encoder = nn.Sequential(*[
            Block(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, images, analytes, vitals, orders, demographics,
                radreport, radreportlabels):
        """
        Returns:
            torch.Tensor: fused embedding [B, output_dim]
        """

        # ---- Encode each modality to 128-dim ----
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
        x = self.input_proj(embeddings)                 # [B, 6, 768]

        cls_tokens = self.cls_token.expand(B, -1, -1)   # [B, 1, 768]
        x = torch.cat((cls_tokens, x), dim=1)           # [B, 7, 768]

        x = x + self.pos_embed[:, :x.size(1), :]        # add positions
        x = self.encoder(x)                             # [B, 7, 768]
        x = self.norm(x)

        cls_out = x[:, 0]                               # [B, 768]
        fused = self.proj(cls_out)                      # [B, 128]

        return F.normalize(fused, p=2, dim=-1)
