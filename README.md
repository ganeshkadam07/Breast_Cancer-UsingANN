# Breast_Cancer-UsingANN
Artificial Neural Networks (ANNs) can be employed for breast cancer detection by analyzing input features such as imaging data and patient information. Trained on a dataset of known outcomes, ANNs learn patterns and relationships to predict the likelihood of breast cancer in new cases. 
# Breast Cancer Prediction using Artificial Neural Networks

This project implements a breast cancer prediction model using Artificial Neural Networks (ANN). The model is trained on a dataset containing various features extracted from breast cancer images.

The goal of this project is to accurately classify breast cancer tumors as malignant (cancerous) or benign (non-cancerous) based on the provided features. The ANN model learns from the dataset to make predictions on new, unseen breast cancer samples.

## Dataset

The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) Dataset, which is publicly available from the UCI Machine Learning Repository. It consists of 569 samples, with 30 features extracted from digitized images of breast mass. Each sample is labeled as either malignant (M) or benign (B).

Link to the dataset: [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

## Model Architecture

The Artificial Neural Network (ANN) used in this project is a feedforward neural network with multiple layers. The architecture of the network is as follows:

- Input Layer: The number of nodes in the input layer is equal to the number of features in the dataset.
- Hidden Layers: The network contains one or more hidden layers with a variable number of nodes. The number of hidden layers and nodes can be configured based on the requirements of the project.
- Output Layer: The output layer consists of a single node that represents the predicted class (malignant or benign).

## Implementation

The breast cancer prediction model is implemented using Python and the Keras library, which is a high-level neural networks API. The following steps are involved in the implementation:

1. Data Preprocessing: The dataset is preprocessed to handle missing values, normalize the features, and encode the labels.
2. Model Training: The ANN model is trained on the preprocessed dataset using backpropagation and gradient descent optimization.
3. Model Evaluation: The trained model is evaluated using various performance metrics such as accuracy, precision, recall, and F1 score.
4. Prediction: The trained model can be used to make predictions on new, unseen breast cancer samples.

## Usage

To run this project locally, follow these steps:

1. Clone the repository:


2. Install the required dependencies:


3. Preprocess the dataset by running the preprocessing script:


4. Train the ANN model by running the training script:


5. Evaluate the model's performance by running the evaluation script:


6. Make predictions using the trained model:

