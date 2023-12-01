RF Formula in Excel & Markdown Latex Equation
```python
height,width,channels = (28,28,1)
pytorch_img = torch.randn(channels,height,width)

conv_operation = nn.Conv2d(in_channels, out_channels, (k,k), stride= 1, padding= 0)
# arguments are in descending order of their importance. (in_channels, out_channels) & (kernel,stride,padding)
# main is kernel size. nested across depth times in block of layers

# padding is choosen for keeping channel size same. Its difficult to manually remember what's the channel size which changes each layer. so better choose a padding value, so that, channel size remains the same. 
```
![Calculations Formula](https://miro.medium.com/v2/resize:fit:744/0*ZJ9F9Sjfxvhvlo6R.)
$$
\begin{align*}
img &= (H,W,C)\\
kernel &= (k,k,C)\\
\\
height_{output} &= \frac {(height_{input} + 2 \cdot padding)}{stride} - \frac{(kernel - 1)}{stride} \tag{1}\\
\end{align*}
$$


$$
\begin{align*}
rf^{global} &= rf_{input} + (kernel - 1) \cdot stride \cdot s_{accum}\\
\\
s_{accum} &= s_{accum} \cdot stride\\
\end{align*}
$$

