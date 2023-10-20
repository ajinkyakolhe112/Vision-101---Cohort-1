import torch, torch.nn as nn, datasets as hg_datasets
import ipyplot

digits_dataset = hg_datasets.load_dataset("mnist", split="train")
split_datasets = digits_dataset.train_test_split(train_size=0.9 , test_size=0.1)

digits_datasets_training, digits_datasets_validation = split_datasets['train'], split_datasets['test']
ipyplot.plot_images(digits_datasets_training['image'][0:5]);

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Define the layers of the model
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28*28*1, 100)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(100, 10)

    def forward(self, x):
        # Flatten the input image
        x = self.flatten(x)

        # Pass the flattened input through the first linear layer
        x = self.linear1(x)

        # Apply the ReLU activation function
        x = self.relu(x)

        # Pass the output of the ReLU layer through the second linear layer
        x = self.linear2(x)

        # Return the output of the second linear layer
        return x


model = Net()
image_tensors, y_actual = digits_datasets_training['image'][0], digits_datasets_training['label'][0]
y_predicted_logits = model.forward(image_tensors)

print(y_predicted_logits)