import torch
import torch.nn as nn


def get_config():
    config = {
        "channels": 3,
        "patch_size": 16,
        "embedding_dim": 768,
        "mlp_size": 3072,
        "num_of_heads": 12,
        "num_of_layers": 12,
        "num_of_classes": 101
    }

    return config

class Embedding(nn.Module):
    def __init__(self, channels, embedding_dim, patch_size, dropout=0.1):
        super.__init__()

        self.dropout = nn.Dropout(dropout)

        # Output shape --> (N, C, H, W)
        self.patch_embedding = nn.Conv2d(
            in_channels=channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        # Output Shape --> (batch_size, embedding_dim, 14*14)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

        batch_size = self.flatten.shape[0]
        embedding_dim = self.flatten.shape[1]
        number_of_patches = self.flatten.shape[2]

        # Output Shape --> (batch_size, 1, embedding_dim)
        self.class_token = nn.Parameter(torch.ones(batch_size, 1, embedding_dim))

        self.position_embedding = nn.Parameter(torch.ones(batch_size, 1+number_of_patches, embedding_dim))


    def forward(self, x):
        # Output Shape --> (batch_size, embedding_dim, H, W)
        x = self.patch_embedding(x)
        # Output Shape --> (batch_size, embedding_dim, 14*14 or H*W)
        x = self.flatten(x)
        # Output Shape --> (batch_size, 14*14 or H*W, embedding_dim)
        x = x.permute(0, 2, 1)

        # Add the class token to patch embedding
        x = torch.cat([self.class_token, x], dim=1)

        # Add positional embedding, Output Shape --> (batch_size, (14*14 or H*W)+1, embedding_dim)
        x = x + self.position_embedding

        # Add dropout as mentioned in B.1
        x = self.dropout(x)

        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super.__init__()

        self.layer_norm = nn.LayerNorm(
            embedding_dim,
        )

        # Dont apply dropout on Multi Head Attention as mentioned in B.1
        self.MultiHeadSelfAttention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True,
        )


    def forward(self, x):
        # Apply Residual connection
        x = x + self.MultiHeadSelfAttention(self.layer_norm(x))

        return x


class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim, dropout=0.1):
        super.__init__()

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(
            embedding_dim,
        )

        self.layer_1 = nn.Linear(
            in_features=embedding_dim,
            out_features=mlp_dim,
        )

        self.gelu = nn.GELU()

        self.layer_2 = nn.Linear(
            in_features=mlp_dim,
            out_features=embedding_dim,
        )


    def forward(self, x):
        y = self.layer_norm(x)
        # Apply 1st Linear layer
        y = self.layer_1(y)
        y = self.dropout(y)
        y = self.gelu(y)

        # Apply 2nd Linear layer
        y = self.layer_2(y)
        y = self.dropout(y)

        # Apply Residual connection
        return x + y


class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim, num_of_classes):
        super.__init__()

        self.layer_norm = nn.LayerNorm(
            embedding_dim,
        )

        self.classification = nn.Linear(
            in_features=embedding_dim,
            out_features=num_of_classes,
        )


    def forward(self, x):
        # only take the first token and pass it through relevant layers as mentioned in 3.1 equation 4
        x = self.layer_norm(x[:, 0, :])

        x = self.classification(x)

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, mlp_dim, dropout=0.1):
        super.__init__()

        self.MultiHeadSelfAttention = MultiHeadSelfAttention(embedding_dim, num_heads)
        self.mlp = MLP(embedding_dim, mlp_dim, dropout)

    def forward(self, x):
        x = self.MultiHeadSelfAttention(x)
        x = self.mlp(x)

        return x


class VisionTransformer(nn.Module):
    def __init__(self, num_of_layers, embedding_dim, num_of_classes, encoder: TransformerEncoderBlock, channels, patch_size, dropout=0.1):
        super.__init__()

        self.embedding = Embedding(channels, embedding_dim, patch_size, dropout)
        self.encoders = nn.Sequential(nn.ModuleList([encoder for x in range(num_of_layers)]))
        self.head = ClassificationHead(embedding_dim, num_of_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoders(x)
        x = self.head(x)

        return x

