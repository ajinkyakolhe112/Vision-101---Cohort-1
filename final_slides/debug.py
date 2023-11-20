import torch, torch.nn as nn
import datasets as huggingface_datasets
from torchvision import datasets as torchvision_datasets
from torchvision import transforms as torchvision_transforms
import ipyplot

BATCH_SIZE = 4

train_dataset    = torchvision_datasets.MNIST( root= '../dataset', transform= torchvision_transforms.ToTensor(), train= True, download= True )
train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])

train_dataloader = torch.utils.data.DataLoader( train_dataset, batch_size = 4, shuffle = True )

TOTAL_BATCHES = len(train_dataset) / BATCH_SIZE


from torch.nn import ReLU as ActivatePositive

MODEL = nn.Sequential(
    nn.Identity(),                                             # LAYER 1: INPUT LAYER
    nn.Flatten(start_dim=1),                                   #          IMAGE RESHAPE
    nn.Linear(out_features = 20, in_features = 28*28*1),     # LAYER 2: 1st Hidden Layer
    ActivatePositive(),                                        #          Activation Function f(x) -> (if x < 0: return 0) & else (if x > 0: return x)
    nn.Linear(out_features = 10 , in_features = 20),         # LAYER 3: Output Layer
)

model_parameters = list(MODEL.parameters())

ERROR_FUNC = nn.functional.cross_entropy
LEARNING_RATE = 0.01
OPTIMIZER  = torch.optim.SGD(params=model_parameters, lr= LEARNING_RATE)

model, error_func, learning_rate, optimizer = MODEL, ERROR_FUNC, LEARNING_RATE, OPTIMIZER

from torchinfo import summary

summary(MODEL, input_size=(1,28,28), 
        verbose=2, col_names = ["input_size", "output_size","kernel_size", "num_params","trainable", "params_percent"]);


import torchmetrics
import wandb
wandb.init()

REPEAT = 10

def trainer_function(train_dataloader, model, error_func, optimizer, epochs):
    model.train(mode=True)
    for epoch_no in range(epochs):

        loss_total, accuracy_total = 0, 0
        for batch_no, (x_actual, y_actual) in enumerate(train_dataloader):

            y_predicted_LOGITS = model(x_actual)
            y_predicted_probs  = nn.functional.softmax(y_predicted_LOGITS, dim= 1)
            loss               = error_func(y_predicted_LOGITS, y_actual)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_batch = loss.item()
            accuracy_batch = torchmetrics.functional.accuracy(y_predicted_LOGITS, y_actual, task="multiclass", num_classes=10)
            
            loss_total = loss_total + loss_batch 
            accuracy_total = accuracy_total + accuracy_batch
            metrics_per_batch = {
                "loss": loss_batch,
                "accuracy_batch": accuracy_batch,
                "batch_no": batch_no
            }
            wandb.log(metrics_per_batch)
            # dError_dWeights = torch.autograd.grad(outputs= loss, inputs = model.parameters() )
            # for parameter, gradient in zip(model.parameters(), dError_dWeights):
            #     parameter = parameter - gradient * learning_rate
        
        accuracy_average = accuracy_total / TOTAL_BATCHES
        metrics_per_epoch = {
            "train_accuracy_epoch": accuracy_average,
            "epoch": epoch_no
        }
        wandb.log(metrics_per_epoch)
        evaluate_model(validation_dataset, model, error_func)

def evaluate_model(evaluation_dataset, model, error_func):
    model.train(mode=False)

    loss_total, accuracy_total = 0, 0
    for x_actual, y_actual in evaluation_dataset:
        x_actual, y_actual = x_actual.unsqueeze(dim=0), torch.tensor(y_actual).unsqueeze(dim=0)
        
        y_predicted_LOGITS = model(x_actual)
        loss = error_func(y_predicted_LOGITS, y_actual)
        accuracy = torchmetrics.functional.accuracy(y_predicted_LOGITS, y_actual, task="multiclass", num_classes=10)
        
        loss_total = loss_total + loss 
        accuracy_total = accuracy_total + accuracy
    
    accuracy_avg = accuracy_total / len(evaluation_dataset)
    wandb.log("validation_accuracy",accuracy_avg)
    

trainer_function(train_dataloader, MODEL, ERROR_FUNC, OPTIMIZER, REPEAT)