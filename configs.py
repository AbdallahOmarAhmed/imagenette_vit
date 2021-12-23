from torchvision.transforms import transforms

dataset_PATH = 'imagenette2-320'  # imagewoof2-320
ImageSize = 320
Mean = (0.5, 0.5, 0.5)
Std = (0.5, 0.5, 0.5)
Augmentation = True
BatchSize = 10
learning_rate = 0.001

model_config = {
    'num_classes': 100,
    'img_size': 32,
    'in_channels': 3,
    'patch_size': 4,
    'embed_dim': 384,
    'depth': 3,
    'num_heads': 4,
    'qkv_bias': True,
    'mlp_ratio': 4,
    'drop_p': 0.25}


def get_aug(train):
    if Augmentation and train:
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize((ImageSize, ImageSize)),
                                   transforms.RandomHorizontalFlip(p=0.5),
                                   transforms.RandomAutocontrast(p=0.5)])
    else:
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize((ImageSize, ImageSize))])

