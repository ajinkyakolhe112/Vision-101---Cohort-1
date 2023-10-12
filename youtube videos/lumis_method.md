

```tex
---
header-includes:
  - \usepackage[ruled,vlined,linesnumbered]{algorithm2e}
---
```

3 Blocks of 2 for Programming

### Block 1
function_get_dataset() -> returns dataset
dataset -> function_get_dataloader () -> returns dataloader

### Block 2
function_get_model_architecture() -> returns model
dataloader, model -> function_get_trainer_of_model() -> returns loss, accuracy
(**AIM** loss : How far the prediction is from the reference answer. & accuracy)

### Block 3
function_plot_graph() -> loss, accuracy graph
function_validation_accuracy() -> loss, accuracy for validation data

---

Maths + Pseducode
$$
x_{actual}, y_{actual} = (x_{[28][28]}, y_{[1]}) \\
model = \Huge f(W,\large <single\_example> \Huge)\\
y_{predicted} = \Huge f(W,\large{x_{actual}} \Huge) \tag{1} \\
loss = y_{predicted} - y_{actual} \\
for each example:\\
	 for each W:\\
	 y_{predicted}
	 loss = y_actual - f(W,)
		W = W - dLoss / dW * 0.001\\
$$

```tex
# Algorithm 1
Just a sample algorithmn
\begin{algorithm}[H]
\DontPrintSemicolon
\SetAlgoLined
\KwResult{Write here the result}
\SetKwInOut{Input}{Input}\SetKwInOut{Output}{Output}
\Input{Write here the input}
\Output{Write here the output}
\BlankLine
\While{While condition}{
    instructions\;
    \eIf{condition}{
        instructions1\;
        instructions2\;
    }{
        instructions3\;
    }
}
\caption{While loop with If/Else condition}
\end{algorithm} 
```