# Mutual Information Neural Estimation (MINE) for Correlated Gaussians

### Structure.
```
├── train_config.yaml
├── mine.py
├── train.py
└── utils.py
└── plots/ (created automatically upon running)
    └── mi_estimation_results.png
````
`train_config.yaml`: Stores all configurable parameters for data generation, model architecture, training, and evaluation.

`mine.py`: Defines the MINE neural network architecture and the EMA (Exponential Moving Average) 

`train.py`: Implements the MINE loss functions (biased and unbiased) and the training loop.

`utils.py`: Provides utility functions.

The script will: load configurations from `train_config.yaml`. Generate synthetic correlated Gaussian data. Train the MINE model for various correlation coefficients. Print the estimated mutual information during training. Generate a plot comparing the true mutual information with the estimated values and save it in the plots/ directory.

### Training
You can modify the `train_config.yaml` file to adjust the parameters.

Launch the training by running:
```
CUDA_VISIBLE_DEVICES=0 python3 train.py
```