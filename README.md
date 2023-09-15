# Privacy Preserving Federated Learning with Convolutional Variational Bottlenecks
This repository contains the implementation of our proposed Convolutional Variational Bottleneck (CVB).

For demonstrations of federated model training follow the federated_training.ipynb Jupyter notebook.
For demonstrations of gradient inversion attacks follow the reconstruction.ipynb Jupyter notebook.

The paper including all empirical results can be found on [arXiv](https://arxiv.org/abs/2309.04515)

## Please cite as:
```
@article{scheliga2023privacy,
  title={Privacy Preserving Federated Learning with Convolutional Variational Bottlenecks},
  author={Scheliga, Daniel and M{\"a}der, Patrick and Seeland, Marco},
  journal={arXiv preprint arXiv:2309.04515},
  year={2023}
}
```

## Abstract:
Gradient inversion attacks are an ubiquitous threat in federated learning as they exploit gradient leakage to reconstruct supposedly private training data. Recent work has proposed to prevent gradient leakage without loss of model utility by incorporating a PRivacy EnhanCing mODulE (PRECODE) based on variational modeling. Without further analysis, it was shown that PRECODE successfully protects against gradient inversion attacks. In this paper, we make multiple contributions. First, we investigate the effect of PRECODE on gradient inversion attacks to reveal its underlying working principle. We show that variational modeling introduces stochasticity into the gradients of PRECODE and the subsequent layers in a neural network. The stochastic gradients of these layers prevent iterative gradient inversion attacks from converging. Second, we formulate an attack that disables the privacy preserving effect of PRECODE by purposefully omitting stochastic gradients during attack optimization. To preserve the privacy preserving effect of PRECODE, our analysis reveals that variational modeling must be placed early in the network. However, early placement of PRECODE is typically not feasible due to reduced model utility and the exploding number of additional model parameters. Therefore, as a third contribution, we propose a novel privacy module -- the Convolutional Variational Bottleneck (CVB) -- that can be placed early in a neural network without suffering from these drawbacks. We conduct an extensive empirical study on three seminal model architectures and six image classification datasets. We find that all architectures are susceptible to gradient leakage attacks, which can be prevented by our proposed CVB. Compared to PRECODE, we show that our novel privacy module requires fewer trainable parameters, and thus computational and communication costs, to effectively preserve privacy.

## Requirements:
+ matplotlib
+ seaborn
+ munch
+ dill
+ prettytable
+ pyyaml
+ numpy
+ pandas
+ pytorch=3.13.1
+ torchvision
+ torchmetrics=0.11.4
+ einops
+ lpips

You can also use [conda](https://www.anaconda.com/) to recreate our virtual environment:
```
conda env create -f environment.yaml
conda activate CVB
```

## Data:
In our experiments we used 6 different classification datasets:
+ [MNIST](https://paperswithcode.com/dataset/mnist)
+ [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
+ [MedMNIST](https://medmnist.com/) (Blood, Derma, Pneumonia, Retina)
  
We provide simple data loading utilities. 
For our reconstruction experiments, victim datasets were sampled randomly from the training datasets. 
We provide these very same victim datasets under *data/victim_datasets*.
More details on how they were samples can be found in the *Experimental Design* section of our paper.