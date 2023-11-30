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
            # loss.backward()
            dE_dW = torch.autograd.grad(outputs = loss, inputs = model.parameters() )
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
            "END OF BATCH"
            # Alternative Training Loop
            
            # loss.backward(), computes dloss/dw for every parameter w which has requires_grad=True.
            # w.grad += dloss/dw. 
            # By default, gradients are accumulated in buffers (i.e, not overwritten) whenever .backward() is called.
            
            # dError_dWeights = torch.autograd.grad(outputs= loss, inputs = model_parameters)
            # for (name, weight), gradient in zip(model.named_parameters(), dError_dWeights):
            #    print(name)
            #    weight = weight - gradient * LEARNING_RATE
        
        accuracy_average    = accuracy_total / TOTAL_BATCHES
        metrics_per_epoch   = {
            "training_accuracy_average_per_epoch": accuracy_average
        }
        wandb.log(metrics_per_epoch)
        "END OF ENTIRE EPOCH no"
        evaluate_model(validation_dataset, model, error_func, epoch_no)

def evaluate_model(dataloader_for_evaluation, model, error_func, epoch_no):
    model.train(mode=False)

    loss_total, accuracy_total = 0, 0
    for x_actual, y_actual in dataloader_for_evaluation:
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