# ! KEY CODE
from torchvision import datasets as torchvision_datasets
from torchvision import transforms as torchvision_transforms
import ipyplot

# 1. Download Dataset from Available once in torchvision_datasets
training_dataset = torchvision_datasets.MNIST( root= '../dataset', transform= torchvision_transforms.ToTensor(), train= True, download= True )
training_dataset, validation_dataset = torch.utils.data.random_split(training_dataset, [0.9, 0.1])

# Plot Images
x,y = training_dataset[0]
ipyplot.plot_images(x);

# Dataset -> Pytorch Tensors
pass

# 2. Pytorch Dataset -> Data Loader
BATCH_SIZE = 4
TOTAL_BATCHES = len(training_dataset) / BATCH_SIZE

training_dataloader   = torch.utils.data.DataLoader( dataset= training_dataset  , batch_size= BATCH_SIZE, shuffle= True )
validation_dataloader = torch.utils.data.DataLoader( dataset= validation_dataset, batch_size= BATCH_SIZE, shuffle= True )
# X_BATCH, Y_BATCH = next(iter(training_dataloader))

# B,C,H,W = BATCH_SIZE,1,28,28
# x_random,y_random = torch.randn(B,C,H,W), torch.randn(1).to(torch.int32)