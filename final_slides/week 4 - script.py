#%%
# ! KEY CODE
from datasets import load_dataset, get_dataset_split_names
import datasets as huggingface_datasets

training_dataset = load_dataset("ajinkyakolhe112/imagenet_10c_tiny", split="train")
# dataset = load_dataset("imagenet-1k")
# get_dataset_split_names("imagenet-1k")
import torchvision, torch
transformations_list = torchvision.transforms.Compose([
        torchvision.transforms.PILToTensor(),
        torchvision.transforms.Resize( size = (256,256) ), 
    ])

def transform_datasets(examples):
    """
        PSEUDO-CODE: `examples['image'] = transforms_list( examples['image'] )`
    """
    assert examples['image'].__len__() == examples['label'].__len__()
    examples['image_tensor']   = []
    for image in examples['image']:
        transformed_image = transformations_list(image)
        transformed_image = transformed_image.to(torch.float32)
        examples['image_tensor'].append(transformed_image)
    
    return examples

training_dataset.set_transform(transform_datasets)

# 1. Pytorch Dataset -> Data Loader
BATCH_SIZE = 10
TOTAL_BATCHES = len(training_dataset) / BATCH_SIZE
training_dataloader   = torch.utils.data.DataLoader( training_dataset  , batch_size= BATCH_SIZE, shuffle= True)

x_batch, y_batch = next(iter(training_dataloader))
#%%

import torch, torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
from torchinfo import summary
summary(model, input_size=(3,256,256), 
        verbose=2, col_names = ["input_size", "output_size","kernel_size", "num_params","trainable", "params_percent"]);
