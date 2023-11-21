# ! KEY CODE

from torchvision import datasets as torchvision_datasets
from torchvision import transforms as torchvision_transforms

# 1. Download Dataset from Available once in torchvision_datasets
training_dataset = torchvision_datasets.MNIST( root= '../dataset', transform= torchvision_transforms.ToTensor(), train= True, download= True )
training_dataset, validation_dataset = torch.utils.data.random_split(training_dataset, [0.9, 0.1])

# 2. Plot Image
x,y = training_dataset[0]
ipyplot.plot_images(x);

# 3. Dataset -> Pytorch Tensors
pass

# 4. Pytorch Dataset -> Data Loader
BATCH_SIZE = 4
TOTAL_BATCHES = len(training_dataset) / BATCH_SIZE

training_dataloader   = torch.utils.data.DataLoader( dataset= training_dataset  , batch_size= BATCH_SIZE, shuffle= True )
validation_dataloader = torch.utils.data.DataLoader( dataset= validation_dataset, batch_size= BATCH_SIZE, shuffle= True )
# X_BATCH, Y_BATCH = next(iter(training_dataloader))
# B,C,H,W = BATCH_SIZE,1,28,28
# x_random,y_random = torch.randn(B,C,H,W), torch.randn(1).to(torch.int32)


from torch.nn import ReLU as ActivatePositive

# 1. Model Architecture - [20 neurons, 10 neurons]
# ! KEY CODE

# network = input -> Linear 20 -> Linear 10 = output
model = nn.Sequential(
    nn.Flatten(start_dim=1), # IMAGE RESHAPE from 2D(28,28) -> 1D(28*28)
    
    nn.Identity(),                                                              # LAYER 1: INPUT LAYER
    nn.Linear(out_features = 20, in_features = 28*28*1), ActivatePositive(),    # LAYER 2: 1st Hidden Layer
    nn.Linear(out_features = 10 , in_features = 20),                            # LAYER 3: Output Layer
)

# 2. Model Parameters & Their Relationship with Error
model_parameters = list(model.parameters())

# 3. Registering Model Parameters with Optimizer, as variables to be minimized
gradient_step    = 0.01
LEARNING_RATE    = gradient_step
OPTIMIZER        = torch.optim.SGD( params= model_parameters, lr= gradient_step )

# 4. Calculation of Error of Prediction
ERROR_FUNC = nn.functional.cross_entropy

# 5. Calculation of relationship of Error & Parameters
X_BATCH, Y_BATCH = next(iter(training_dataloader))
GRADIENTS_accumulated = torch.autograd.grad(outputs = ERROR_FUNC(model(X_BATCH), Y_BATCH), inputs = model_parameters)
# loss.backward(), computes dloss/dw for every parameter w which has requires_grad=True.
# w.grad += dloss/dw. By default, gradients are accumulated in buffers (i.e, not overwritten) whenever .backward() is called.

# 6. Model Layers, Parameters Visualization
from torchinfo import summary
summary(model, input_size=(1,28,28), 
        verbose=2, col_names = ["input_size", "output_size","kernel_size", "num_params","trainable", "params_percent"]);

# ! KEY CODE

import torchmetrics
import wandb
wandb.init()

REPEAT = 10

def trainer_function(training_dataloader, model, error_func, optimizer, epochs):
    model.train(mode=True)

    for epoch_no in range(epochs):

        loss_total, accuracy_total = 0, 0
        for batch_no, (x_actual, y_actual) in enumerate(training_dataloader):

            y_predicted_LOGITS = model.forward(x_actual)
            y_predicted_probs  = nn.functional.softmax(y_predicted_LOGITS, dim= 1)
            loss               = error_func(y_predicted_LOGITS, y_actual)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_batch      = loss.item()
            accuracy_batch  = torchmetrics.functional.accuracy(y_predicted_LOGITS, y_actual, task="multiclass", num_classes=10)
            
            loss_total      = loss_total + loss_batch 
            accuracy_total  = accuracy_total + accuracy_batch
            
            metrics_per_batch = {
                "loss": loss_batch,
                "accuracy_batch": accuracy_batch,
                "accuracy_average": accuracy_total / (batch_no + 1),
                "batch_no": batch_no
            }
            wandb.log(metrics_per_batch)
            print("END OF BATCH")
            """
            # Alternative Training Loop
            
            OPTIMIZER.zero_grad()
            # loss.backward()
            # loss.backward(), computes dloss/dw for every parameter w which has requires_grad=True.
            # w.grad += dloss/dw. By default, gradients are accumulated in buffers (i.e, not overwritten) whenever .backward() is called.
            dError_dWeights = torch.autograd.grad(outputs= loss, inputs = model_parameters)

            # SINGLE STEP UPDATES ALL PARAMETERS by one STEP
            OPTIMIZER.step()
            for (name, weight), gradient in zip(model.named_parameters(), dError_dWeights):
                print(name)
                weight = weight - gradient * LEARNING_RATE
            """
            print("END OF BATCH")
        
        accuracy_average    = accuracy_total / TOTAL_BATCHES
        metrics_per_epoch   = {
            "training_accuracy_average_per_epoch": accuracy_average
        }
        wandb.log(metrics_per_epoch)
        print(f"END OF ENTIRE EPOCH no {epoch_no}")
        evaluate_model(validation_dataset, model, error_func, epoch_no)

def evaluate_model(dataset, model, error_func, epoch_no):
    model.train(mode=False)

    loss_total, accuracy_total = 0, 0
    for x_actual, y_actual in validation_dataloader:
        y_predicted_LOGITS = model(x_actual)
        loss = error_func(y_predicted_LOGITS, y_actual)
        accuracy = torchmetrics.functional.accuracy(y_predicted_LOGITS, y_actual, task="multiclass", num_classes=10)
        
        loss_total = loss_total + loss 
        accuracy_total = accuracy_total + accuracy
    
    accuracy_average = accuracy_total / len(dataset)
    wandb.log({
        "validation_accuracy_average_per_epoch": accuracy_average
        })

trainer_function(training_dataloader, model, ERROR_FUNC, OPTIMIZER, REPEAT)