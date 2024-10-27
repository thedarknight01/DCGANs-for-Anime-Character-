
# DCGAN for Anime Character Generation

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate anime characters. The model is built using PyTorch and is designed to create high-quality images from random noise inputs.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Generative Adversarial Networks (GANs) are a class of machine learning frameworks designed to generate new data instances that resemble a training dataset. This project focuses on DCGANs, which use deep convolutional networks for both the generator and discriminator.

## Features
- Generate high-resolution anime character images.
- Modular architecture for easy modifications.
- Support for custom datasets.

## Requirements

- Python >= 3.7
- PyTorch >= 1.10.0
- torchvision >= 0.11.1
- numpy
- matplotlib

You can install the required packages using:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/thedarknight01/DCGANs-for-Anime-Character-.git
   cd DCGANs-for-Anime-Character-
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the DCGAN model, use the following command:

```bash
python training.py --num_epochs 50 --batch_size 64 --learning_rate 0.0002
```

Replace the arguments with your desired values.

### Generating Images

To generate images using a pre-trained model, run:

```bash
python generate.py --model_path models/dcgan.pth --num_images 10
```

This will create 10 images using the trained DCGAN model.

## Training

This model was trained on *animefacedataset by splcher* which can be found on [Kaggle](https://www.kaggle.com/datasets/splcher/animefacedataset). Training may take several hours depending on your hardware. For best results, use a GPU.

## Results

![Generated Image Example](/Dcgans.png)


## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

##### Keep Doing Better :)
##### Feel free to share any ideas for improvement and also check out the dataset owner on Kaggle
