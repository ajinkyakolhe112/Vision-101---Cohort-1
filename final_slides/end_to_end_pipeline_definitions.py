import torch

def get_datasets():
    training_dataset, validation_dataset = torch.utils.data.Dataset, torch.utils.data.Dataset
    return training_dataset, validation_dataset

def get_dataloaders(training_dataset, validation_dataset):
    training_dataloader, validation_dataloader = torch.utils.data.DataLoader, torch.utils.data.DataLoader
    return training_dataloader, validation_dataloader

def get_untrained_model():
    model = torch.nn.Sequential()
    return model
def train_model(model, training_dataloader, validation_dataloader):

    trained_model = model
    return trained_model

"""
    error_function = nn.functional.cross_entropy
    gradient_step_length = 0.001
    parameters_to_optimize = list(model.named_parameters())
    optimizer = torch.optim.SGD(params = parameters_to_optimize, lr = gradient_step_length)

    for batch_no, (x_actual, y_actual) in enumerate(training_dataloader):
        y_pred_logits  = model.forward(x_actual)
        loss                = torchmetrics.functional.accuracy(y_predicted_logits, y_actual)

        optimizer.zero_grad()
        dE_dW = torch.autograd.grad(loss, parameters_to_optimize)
        for name, parameter, gradient in zip(parameters_to_optimize, dE_dW):
            parameter =  parameter - gradient * gradient_step_length

            print(name)
"""