import torch, torch.nn as n

from end_to_end_pipeline_definitions import get_datasets, get_dataloaders, get_untrained_model, train_model

training_dataset,    validation_dataset     = get_datasets()
training_dataloader, validation_dataloader  = get_dataloaders(training_dataset, validation_dataset)

model          = get_untrained_model()
trained_model  = train_model(model, training_dataloader, validation_dataloader)
