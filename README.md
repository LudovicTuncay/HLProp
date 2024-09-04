# Hierarchical Label Propagation (HLP)
Implementation of Hierarchical Label Propagation (HLP) in python.

## Installation
This code is included in the `hierarchical-label-propagation` package. To install it, you can run the following command:

```bash
pip install hierarchical-label-propagation
```

## Usage
The `hierarchical-label-propagation` package provides a class called `HLP` that can be used to run the Hierarchical Label Propagation algorithm. The following code snippet shows how to use it:

```python
from hierarchical_label_propagation import HLP

hlp = HLP() # Creates an instance with the default HLP method for AudioSet.
``` 

HLP can now be applied to the AudioSet targets. Using the same API, it can either be done to the target of a single example or to the targets of a batch of examples. The following code snippet shows how to apply HLP:

```python
y = hlp.propagate(y) # Propagates the labels of a single example.
```
or
```python
X, Y, file_names = batch
Y = hlp.propagate(Y) # Propagates the labels of a batch of examples.
```

Please note that hlp.propagate() takes a 1D (# classes) or 2D (Batch Size, # classes) torch.Tensor as an input and returns the propagated labels as a torch.Tensor of the same shape as the input.

the `propagate` method can also be used to propagate on continuous values using the same API. This is true if and only if a higher value indicates a higher confidence in the label. This can be useful to apply HLP to the output of a classifier as a post-processing step.

## Citation
If you use this code in your research, please consider citing the following paper:

```
Citation will be added soon.
```