from datasets import load_dataset

# Load the MNIST dataset
mnist_dataset = load_dataset("mnist")

import torch

class MNISTModel(torch.nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

from transformers import TrainingArguments

# Define training hyperparameters
training_args = TrainingArguments(
    output_dir="my_mnist_model",
    num_train_epochs=10,
    learning_rate=1e-3,
    weight_decay=0.01,
)

from transformers import Trainer

# Instantiate the Hugging Face Trainer API
trainer = Trainer(
    model=MNISTModel(),
    args=training_args,
    train_dataset=mnist_dataset["train"],
    eval_dataset=mnist_dataset["test"],
)
# Train the model
trainer.train()
