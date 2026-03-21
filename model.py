import torch
import torch.nn as nn
from typing import List


class NCF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        gmf_embed_dim: int = 32,
        mlp_embed_dim: int = 32,
        mlp_layer_sizes: List[int] = [256, 128, 64],
        dropout: float = 0.2,
    ):
        """
        Neural Collaborative Filtering model combining GMF and MLP branches.

        Args:
            num_users:        Total number of users (from metadata.json)
            num_items:        Total number of items (from metadata.json)
            gmf_embed_dim:    Embedding dimension for the GMF branch
            mlp_embed_dim:    Embedding dimension for the MLP branch
            mlp_layer_sizes:  List of hidden layer sizes for the MLP branch
            dropout:          Dropout rate applied after each MLP hidden layer
        """
        super(NCF, self).__init__()

        # --- GMF branch embeddings ---
        self.gmf_user_embedding = nn.Embedding(num_users, gmf_embed_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, gmf_embed_dim)

        # --- MLP branch embeddings (separate from GMF) ---
        self.mlp_user_embedding = nn.Embedding(num_users, mlp_embed_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, mlp_embed_dim)

        # --- MLP branch: concatenated embeddings -> FC layers with ReLU + Dropout ---
        mlp_layers = []
        input_size = 2 * mlp_embed_dim  # user + item concatenated
        for output_size in mlp_layer_sizes:
            mlp_layers.append(nn.Linear(input_size, output_size))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(p=dropout))
            input_size = output_size
        self.mlp = nn.Sequential(*mlp_layers)

        # --- Fusion + output ---
        # GMF contributes gmf_embed_dim, MLP contributes last layer size
        fusion_input_size = gmf_embed_dim + mlp_layer_sizes[-1]
        self.output_layer = nn.Linear(fusion_input_size, 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_ids: LongTensor of shape (batch_size,)
            item_ids: LongTensor of shape (batch_size,)
        Returns:
            FloatTensor of shape (batch_size,) with sigmoid probabilities
        """
        # GMF branch: element-wise product
        gmf_user = self.gmf_user_embedding(user_ids)
        gmf_item = self.gmf_item_embedding(item_ids)
        gmf_out = gmf_user * gmf_item

        # MLP branch: concatenate then pass through FC layers
        mlp_user = self.mlp_user_embedding(user_ids)
        mlp_item = self.mlp_item_embedding(item_ids)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_out = self.mlp(mlp_input)

        # Fusion: concatenate GMF and MLP outputs -> sigmoid
        fusion = torch.cat([gmf_out, mlp_out], dim=-1)
        out = self.sigmoid(self.output_layer(fusion))
        return out.squeeze(-1)
