import torch
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim import Adam
from configs import learning_rate, accuracy, maxAcc, BatchSize
from transformerblock import PatchEmbedding, AttnBlock


class VisionTransformer(LightningModule):
    def __init__(self,
                 img_size=320,
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
        pred = self(x)
        loss = self.Loss(pred, y)
        train_acc = accuracy(pred, y)
        self.log('train loss', loss/BatchSize)
        self.log('train acc', train_acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.Loss(pred, y)
        # import ipdb;ipdb.set_trace()
        test_acc = accuracy(pred, y)
        self.log('test loss', loss/BatchSize)
        self.log('test acc', test_acc)
        if test_acc > maxAcc:
            torch.save(self.state_dict(), 'model.pth')

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
