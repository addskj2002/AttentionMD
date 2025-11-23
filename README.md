# Repository for "Optimizing Attention with Mirror Descent: Generalized Max-Margin Token Selection" paper

This repository contains the codebase for the experiments done within the paper "Optimizing Attention with Mirror Descent: Generalized Max-Margin Token Selection". The library requirements is as followed
```
torch==2.0.0
numpy==1.24.3
transformers==4.41.2
```

The `real-data` folder contains the code for the experiments with the data from the [Stanford Large Movie Dataset](https://ai.stanford.edu/~amaas/data/sentiment/), CIFAR-10, and CIFAR-100, while `synthetic-data` contains experiments with randomly (with seeds) generated data.
