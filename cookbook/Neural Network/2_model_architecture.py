
# ! KEY CODE
import torch, torch.nn as nn
from torch.nn import ReLU as ActivatePositive

# 1. Model Architecture - [20 neurons, 10 neurons]
# network = input -> Linear 20 -> Linear 10 = output
model = nn.Sequential(
    nn.Flatten(start_dim=1), # IMAGE RESHAPE from 2D(28,28) -> 1D(28*28)
    
    nn.Identity(),                                                              # LAYER 1: INPUT LAYER
    nn.Linear(out_features = 20, in_features = 28*28*1), ActivatePositive(),    # LAYER 2: 1st Hidden Layer
    nn.Linear(out_features = 10 , in_features = 20),                            # LAYER 3: Output Layer
    # NO ACTIVATION FUNCTION ON FINAL LAYER. Pre activation value = `logits`
)

# 2. Model Parameters & Their Relationship with Error
model_parameters = list(model.parameters())
# 3. Model Layers, Parameters Visualization
from torchinfo import summary
summary(model, input_size=(1,28,28), 
        verbose=2, col_names = ["input_size", "output_size","kernel_size", "num_params","trainable", "params_percent"]);

# ! FOR A SINGLE BATCH
X_BATCH, Y_BATCH = next(iter(training_dataloader))

# 4. Registering Model Parameters with Optimizer, as variables to be minimized
gradient_step    = 0.01
LEARNING_RATE    = gradient_step
OPTIMIZER        = torch.optim.SGD( params= model_parameters, lr= gradient_step )
# 5. Calculation of Error of Prediction
ERROR_FUNC = nn.functional.cross_entropy
# 6. Calculation of relationship of Error & Parameters
dError_dParameters = torch.autograd.grad(outputs = ERROR_FUNC(model(X_BATCH), Y_BATCH), inputs = model_parameters)
# loss.backward(), computes dloss/dw for every parameter w which has requires_grad=True.
# w.grad += dloss/dw. By default, gradients are accumulated in buffers (i.e, not overwritten) whenever .backward() is called.