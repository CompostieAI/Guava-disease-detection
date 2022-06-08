# Device friendly Guava Fruits and leave disease detection using deep learnin

Fruits and leave disease detection using image analysis is an important research problem in smart agriculture. In this work, we have investigated several image classification techniques to detect Guava disease from fruits and leave images. We directly classify images  by ignoring segmentation the affected regions of the images. The experimental performances show more that 95\% accuracy for most of the state-of-art convolution based methods. We also explore model size reduction techniques to use the model at mobile devices instead of use cloud prediction system. We are able to compress the model in a reasonable size without degrading the performance of the model. Our works provides an analysis of model performance and size reduction to use the model at mobile application.

## Installation

Several libraries are needed to be installed for training to work. I will assume that everything is being installed in an Anaconda installation on Ubuntu, with PyTorch installed.

Install PyTorch if you haven't already.And also install all libraries mentioned in requirements.txt

If you want to setup from this repository then run bellow command:
```bash
git colne https://github.com/CompostieAI/Guava-disease-detection.git
cd Guava-disease-detection

```
Finaly run the following command to ensure that everything is being installed correctly.
```bash
pip install -r requirements.txt
```

# Datasets

The dataset is based on Bangladesh and collected from a large Guava garden in the middle of 2021 by an expert team of Bangladesh Agricultural University.
No pretreatment was applied before collecting the image using a Digital SLR Camera [see references].

The dataset consists of four typical diseases: 

   * Phytophthora
   * Red Rus
   * Scab
   * Styler end Rot 
   * Disease-free leaf and fruits


Download the CelebA dataset and put images and attributes text file in dataset folder.
* Dataset
  * [Guava Leaves and Fruits Dataset for Guava Disease](https://data.mendeley.com/datasets/x84p2g3k6z/1) dataset
   

# System Overview

<p align="center">
  <img src="examples/system_overview.png" align="center" width="1000" height="400" />
</p>


# Preprocess

Make different folder as per class name and put all images both real and augmented in the corresponding folder.


# Training

Run the follwing command to train the model. It trains on training dataset and save checkpoint 

```bash
python train.py --root_path "your-root-path-toproject-directory" --image_container_path "your path" --checkpoint_path "your_path" 

Example:
   python train.py --root_path '/home/incentive/Music/samrt_agriculture' --image_container_path "data/guava" --checkpoint_path 'models/' 

```
# Save model

After training model put the saved model weights in weights folder.
```bash
mkdir weights
```
# Test

Run the follwing command to test AttGAN model on test dataset.This will generate attribute wise images from test dataset.

```bash
python test.py --root_path "your-root-path-toproject-directory" --weights_path "name of model weight"

Example: 
   python test.py --root_path '/home/incentive/Music/samrt_agriculture' --weights_path 'src/weights/weights.149.pth'

```

# Inference

After running the inference file it will class name of that given input image.

```bash
python inference.py - --root_path "your-root-path-toproject-directory" --weights_path "name of model weight"

Example: 
   python test.py --root_path '/home/incentive/Music/samrt_agriculture' --weights_path 'src/weights/weights.149.pth'
```


# Optimization
Go to the optimization folder and follow the instruction from the readme file.


# Extras

If you want to know about specific files or its function then follow the following command

* file information
```bash
python filename.py --help
```

* function information
```bash
python #goto python bash

from filename import *
help(function_name)

```

# References
* [https://data.mendeley.com/datasets/x84p2g3k6z/1] (https://data.mendeley.com/datasets/x84p2g3k6z/1)
* [Optimize machine learning models](https://www.tensorflow.org/model_optimization)