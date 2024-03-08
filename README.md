# Sarcasm Detection using Neural Networks

This Python script demonstrates how to build and train a neural network for sarcasm detection using TensorFlow and Keras. The script downloads a dataset of sarcastic and non-sarcastic headlines, preprocesses the data, tokenizes the text, pads sequences, and trains a neural network model.

## Dataset

The dataset used in this script is a collection of headlines labeled as sarcastic or non-sarcastic. It is downloaded from a [Google Cloud Storage URL](https://storage.googleapis.com/learning-datasets/sarcasm.json) and saved as a JSON file.

## Data Preprocessing

The script preprocesses the headlines by tokenizing them, converting them to sequences, and padding the sequences to ensure uniform length. The dataset is split into training and testing sets.

## Model Architecture

The neural network model consists of an embedding layer, a global average pooling 1D layer, a dense layer with ReLU activation, and a final dense layer with sigmoid activation for binary classification.

## Training

The model is compiled with the Adam optimizer and binary crossentropy loss function. It is trained for 30 epochs on the training set and evaluated on the testing set.

## Visualization

The script uses matplotlib to plot the training and validation accuracy and loss curves.

## Word Embeddings

The script extracts word embeddings from the embedding layer of the trained model and saves them to files (`vecs.tsv` and `meta.tsv`). These files can be used for visualizing word embeddings in tools like TensorBoard.

## Prediction

The script includes an example of predicting sarcasm for given sentences using the trained model.

## Requirements

- Python
- TensorFlow
- Keras
- Matplotlib
- Requests

## Usage

1. Run the script to download the dataset and train the model.
2. Use the trained model for sarcasm detection on new sentences.
