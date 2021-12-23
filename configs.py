import torch
from torchvision.transforms import transforms

dataset_PATH = 'imagenette2-320'  # imagewoof2-320
Mean = (0.5, 0.5, 0.5)
Std = (0.5, 0.5, 0.5)
BatchSize = 100
ImageSize = 320
Augmentation = True
learning_rate = 0.001
maxAcc = 0

model_config = {
    'num_classes': 10,
    'img_size': 320,
    'in_channels': 3,
    'patch_size': 16,
    'embed_dim': 384,
    'depth': 4,
    'num_heads': 4,
    'qkv_bias': True,
    'mlp_ratio': 4,
    'drop_p': 0.}


def get_aug(train):
    if Augmentation and train:
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize((ImageSize, ImageSize)),
                                   transforms.RandomHorizontalFlip(p=0.5),
                                   transforms.RandomAutocontrast(p=0.5)])
    else:
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize((ImageSize, ImageSize))])


def accuracy(preds, y):
    _, y_hat = torch.max(preds, dim=1)
    correct = (y_hat == y).float()
    acc = correct.sum() / len(correct)
    return acc

# {
#     'num_classes': 10,
#     'img_size': 320,
#     'in_channels': 3,
#     'patch_size': 16,
#     'embed_dim': 384,
#     'depth': 4,
#     'num_heads': 4,
#     'qkv_bias': True,
#     'mlp_ratio': 4,
#     'drop_p': 0.,
#     'ImageSize': 320,
#     'Augmentation': True,
#     'learning_rate': 0.001
# }