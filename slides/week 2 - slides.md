
## Agenda
1. Intelligence


---
## Intelligence in Brain 
- Diagram with subparts. General Intelligence with multiple different sub neural networks working together. 
At least 40 sub neural networks. 
- AGN + 40 different functional neural networks at least
We don't have to follow the structure of brain. We can come up with a better one, but its a good starting point. 

---
## Complexity of Vision
- Very long distance and important information
- But bandwidth is huge. You can check youtube video vs 1 text web page data usage
- Complexity of scenes is huge. even 32*32 pixels problem takes very long to learn. We need at least 10^6 * 10^6 pixels (1 MP) 
    - Callback to cifar10 homework assignment
- Color shades to detect = 256 per color, Colors to detect, objects to detect infinite
- Like our brain we, need to be able to process the volume efficiently. So better brute force than last week
---
## Last Week's Status
```python
feature_extractor = nn.Sequential( # build a function which returns model built from list of channels.
  nn.Conv2d(1,50,3),
  nn.ReLU(),
  nn.BatchNorm2d(50),
  
  nn.Conv2d(1,50,3),
  nn.ReLU(),
  nn.BatchNorm2d(50),
  

)
decision_maker = nn.Sequential(
  nn.Linear(500,50)
  nn.Linear(50,10)
)
model = nn.Sequential(
  feature_extractor,
  decision_maker
)
y_predicted_logits = model(example)
y_predicted_probs  = nn.functional.softmax(y_predicted_logits)

```

---
## Visual Cortex
- Structure of visual cortex. [eye -> optic nerve -> lgn -> v1 -> v2 -> v3 -> v4 ]
- Hypothetical Grandmother neuron
- 4 Blocks of Feature Extractor
---
## Understanding Image Channels & Filters

---
## Model Parameters in Detail

---
## Intelligence on Any Data
- data from eye -> electrical pulse
- data from nose -> electrical pulse
- data from taste, touch everything -> its an electrical pulse
- Last slide we said, artificial nn, work on any data for any problem
- We have so many different animals, all with different locations. monkey, fish, bird... different umvelt... 


---
## Convolution & Multi-Channel Convolution
- Convolution Explaining Video
---
## Filter what Does it Extract

---
## Filter Visualization & Flexibility of Artificial
- Can see what kind of images maximally activate
- Can see what kind of images maximally deactivate
- Can recognize & customize it to do create not just classification. 
---
- Write start to end neural network, from thought process live.