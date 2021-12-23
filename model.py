import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam
from configs import learning_rate


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # LP = linear projection
        self.LP = nn.Conv2d(in_channels, embed_dim,
                            kernel_size=patch_size, stride=patch_size)

    def forward(self, x):  # (num_samples, in_channels, img_size, img_size)
        out = self.LP(x)  # (num_samples, embed_dim, sqrt(num_patches)), sqrt(num_patches))
        out = out.flatten(2)  # (num_samples, embed_dim, num_patches)
        out = out.transpose(1, 2)  # (num_samples, num_patches, embed_dim)
        return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, drop_p=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** (-0.5)

        self.qkv_layer = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(drop_p)
        self.proj_layer = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop_p)

    def forward(self, x):  # (num_samples, num_patches+1, dim)
        num_samples, num_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv_layer(x)  # (num_samples, num_patches+1, dim*3)
        qkv = qkv.reshape(num_samples, num_tokens, 3,
                          self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # qkv >>> (3, num_samples, num_heads, num_tokens, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k_t = k.transpose(-2, -1)
        # DP = dot product
        DP = (q @ k_t) * self.scale  # (num_samples, num_heads, num_tokens, num_tokens)
        attn = DP.softmax(dim=-1)
        attn = self.attn_drop(attn)

        A = attn @ v
        A = A.transpose(1, 2)  # (num_samples, num_tokens, num_heads, head_dim)
        A = A.flatten(2)  # (num_samples, num_tokens, dim)

        out = self.proj_layer(A)
        out = self.proj_drop(out)
        return out


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop1 = nn.Dropout(p)
        self.drop2 = nn.Dropout(p)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.drop2(out)
        return out


class AttnBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            drop_p=drop_p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim)

    def forward(self, x):
        out = x + self.attn(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class VisionTransformer(LightningModule):
    def __init__(self,
                 img_size=384,
                 patch_size=16,
                 in_channels=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_p=0.):
        super().__init__()
        self.patch_embedding = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1+self.patch_embedding.num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            AttnBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_p=drop_p)
            for _ in range(depth)
            ])
        self.pos_drop = nn.Dropout(p=drop_p)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)
        self.Loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        num_samples = x.shape[0]
        out = self.patch_embedding(x)
        cls_token = self.cls_token.expand(num_samples, -1, -1)
        out = torch.cat((cls_token, out), dim=1)
        out = out + self.pos_embed
        out = self.pos_drop(out)
        for block in self.blocks:
            out = block(out)
        out = self.norm(out)
        cls_token_end = out[:, 0]
        out = self.head(cls_token_end)
        return out

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.Loss(logits, y)
        return loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
