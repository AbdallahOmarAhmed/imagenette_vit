from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from configs import dataset_PATH, get_aug, BatchSize, model_config
from torchvision.datasets import ImageFolder
from os.path import join
from model import VisionTransformer

# loading data set
train_aug = get_aug(True)
test_aug = get_aug(False)
train_DataSet = ImageFolder(join(dataset_PATH, 'train'), transform=train_aug)
test_DataSet = ImageFolder(join(dataset_PATH, 'val'), transform=test_aug)
train_dataloader = DataLoader(train_DataSet, batch_size=BatchSize, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_DataSet, batch_size=BatchSize, shuffle=False, num_workers=4)
print('finished loading dataset')

# train
model = VisionTransformer(**model_config)
trainer = Trainer(gpus=1, precision=16)
trainer.fit(model, train_dataloader, test_dataloader)
