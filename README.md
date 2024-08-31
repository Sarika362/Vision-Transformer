# Vision-Transformer 🖼️ for CIFAR-10 Image Classification

This project showcases a Vision Transformer (ViT) model, a deep learning architecture originally designed for Natural Language Processing (NLP) but adapted here for image recognition tasks. The model is applied to the CIFAR-10 dataset, enabling the classification of images into 10 different categories.

## Project Overview 📋

The Vision Transformer (ViT) model is a transformer-based deep learning architecture that processes images by breaking them down into smaller patches, applying transformer layers, and aggregating the information for image classification. This project includes the following steps:

1. **Dataset Preparation**: Load and preprocess the CIFAR-10 dataset.
2. **Data Augmentation**: Apply augmentation techniques to the training data.
3. **Model Architecture**: Construct the Vision Transformer (ViT) model using TensorFlow and Keras.
4. **Training**: Train the model using the CIFAR-10 dataset.
5. **Evaluation**: Evaluate the model's accuracy on test data.
6. **Prediction**: Predict and visualize the classification of images from the test set.


## Features 🌟

- **Data Handling**: Load and preprocess the CIFAR-10 dataset.
- **Data Augmentation**: Utilize Keras for augmenting training data.
- **Vision Transformer Model**: Implement a Vision Transformer for image classification.
- **Training & Evaluation**: Train the model and evaluate its performance on test data.
- **Visualization**: Display the images and their predicted labels.

## Installation 🛠️

1. Clone the repository:

    ```bash
    git clone https://github.com/Sarika362/Vision-Transformer.git
    ```

2. Navigate to the project directory:

    ```bash
    cd Vision-Transformer
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage 🚀

1. Run the project:

    ```bash
    python app.py
    ```

2. View the output: The model will display images from the CIFAR-10 dataset and predict their classes.

## Dataset 📚

**CIFAR-10**: The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes. The dataset is divided into 50,000 training images and 10,000 test images.

## Model Architecture 🏗️

The Vision Transformer (ViT) model is built using the following key components:

- **Patches Layer**: Breaks down the input images into smaller patches.
- **Patch Encoder Layer**: Encodes the patches with positional embeddings.
- **Transformer Layers**: Applies multiple layers of transformers to process the encoded patches.
- **MLP Head**: Classifies the transformed patches into one of the 10 categories.

## Results 📊

After training the Vision Transformer model, the performance is evaluated on the test dataset, and the accuracy metrics are reported. The model achieves competitive accuracy on the CIFAR-10 dataset.

## Example Output 🖼️

Below are examples of the model's predictions on images from the CIFAR-10 test set:

- **Image 1**: Class: Dog
- **Image 2**: Class: Airplane


## Acknowledgements 🙏

- The CIFAR-10 dataset is provided by the Canadian Institute for Advanced Research.
- TensorFlow and Keras libraries are used for building and training the model.
