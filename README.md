# Code accompanying "Better Training using Weight-Constrained Stochastic Dynamics", to appear in ICML 2021

In this work we use constrained stochastic differential equations to train neural networks.
As specific constraints we consider circle and orthogonality constraints.
The corresponding optimizers can be found in the Optimizer folder.

Experiments provided here are:

- Our orthogonal constraint method (o-CoLA-ud) for a ResNet-34 architecture with learning rate decay on CIFAR-10 data (Figure 6 in the paper).
 
    python OGconstraint_CIFAR10_resnet34.py
  
- Our circle constraint method (c-CoLA-ud) for a MLP architecture on Fashion-MNIST data with 10K training samples (Figure 7 in the paper).

    python circleconstraint_FashionMNIST_MLP.py


