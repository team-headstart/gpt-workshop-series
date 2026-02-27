# Headstarter GPT Workshop Series

<p align="center">
  <img src="GPT-Workshop-Series-Img.png" alt="GPT Workshop Series Banner" width="100%"/>
</p>

A hands-on workshop series building up to GPT from scratch, following Andrej Karpathy's Neural Networks: Zero to Hero course. Organized by [Headstarter](https://www.headstarter.co) and led by [Saad Jamal](https://www.linkedin.com/in/sadjamz/).

## Overview

This series walks through the core concepts of neural networks step by step: starting from what a derivative is and ending with training language models. Each lecture is a self-contained Jupyter notebook with code, explanations, and visualizations.

## Lectures

### Lecture 1: Micrograd & Backpropagation
`lecture1.ipynb`

- **Derivatives from first principles**: limit definition, slopes, and signs
- **Value object**: building an autograd scalar with operator overloading (`+`, `*`, `tanh`, `exp`, `pow`)
- **Computation graphs**: visualizing the DAG with Graphviz
- **Chain rule & backpropagation**: manually computing gradients, then automating it with topological sort
- **Single neuron**: forward pass through `w * x + b` with tanh activation
- **Multi-layer perceptron**: `Neuron`, `Layer`, and `MLP` classes from scratch
- **Training loop**: forward pass, MSE loss, backward pass, gradient descent
- **PyTorch comparison**: verifying gradients match PyTorch's autograd

### Lecture 2: Bigram Language Model
`lecture2.ipynb`

- **Character-level language modeling**: predicting the next character from the previous one
- **Bigram statistics**: counting character pair frequencies from a 32K name dataset
- **Probability distributions**: normalizing counts, sampling with `torch.multinomial`
- **Broadcasting semantics**: practical exercises with PyTorch tensor operations
- **Maximum likelihood estimation**: log likelihood, negative log likelihood as a loss function
- **Smoothing**: handling zero-count bigrams with additive smoothing
- **Neural network approach**: one-hot encoding, logits, softmax, and gradient descent to learn the same bigram model
- **Regularization**: penalizing large weights to produce smoother distributions
- **Teaser for next lecture**: building the dataset for a trigram / MLP model (Bengio et al., 2003)

## Getting Started



## Resources

- [Neural Networks: Zero to Hero (YouTube)](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ): Andrej Karpathy
- [micrograd](https://github.com/karpathy/micrograd): Karpathy's autograd engine
- [A Neural Probabilistic Language Model (Bengio et al., 2003)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
