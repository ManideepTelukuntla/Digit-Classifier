# Digit-Classifier

<div align="center">
  <img src="https://github.com/ManideepTelukuntla/Digit-Classifier/blob/main/Images/Digit-Classifier-Interface.png" width="800" height="425" alt="Digit Classifier Interface">
  <br>
  <p>Digit Classifier Interface <a href="https://msbaoptim2-4.anvil.app/">Webpage</a></p>
</div>

## Table of Contents
1. [Introduction/Overview](#1-introductionoverview)
2. [Project Goals](#2-project-goals)
3. [Architecture Overview](#3-architecture-overview)
4. [Installation/Requirements](#4-installationrequirements)
5. [File Descriptions](#5-file-descriptions)
6. [Data Pre-processing](#6-data-pre-processing)
7. [Model Implementation](#7-model-implementation)
8. [Analysis and Recommendations](#8-analysis-and-recommendations)
9. [Conclusion](#9-conclusion)
10. [License](#10-license)

## 1. Introduction/Overview
Digit Classifier is a robust digit prediction system utilizing CNN &amp; ViT models on the MNIST dataset, featuring a Flask backend on AWS ECS with Docker and an intuitive Anvil UI. Perfect for scalable, accurate digit recognition.

## 2. Project Goals
- **Development of a Web-Based Platform**: Aiming for a seamless integration of advanced machine learning models within a user-friendly web application, enabling users to upload digit images for quick predictions.
- **Achieving High Accuracy in Model Predictions**: Focusing on fine-tuning CNN and ViT models to achieve or surpass a benchmark accuracy of 99% on the MNIST validation dataset.
- **Interpretation and Improvement of Model Performance**: Committing to the analysis of both correctly and incorrectly classified digits to enhance the models' accuracy and reliability.

## 3. Architecture Overview
This project combines a user-friendly web interface developed with Anvil, a robust backend powered by Flask, and a scalable deployment on AWS ECS using Docker containers. At its core, it employs CNN and ViT models, trained on the MNIST dataset, for accurate and efficient digit prediction.

<div align="center">
  <img src="https://github.com/ManideepTelukuntla/Digit-Classifier/blob/main/Images/Framework.png" width="800" height="254" alt="Digit Classifier Framework">
  <br>
  <p>Digit Classifier Framework</p>
</div>

## 4. Installation/Requirements
- **Python & TensorFlow**: Utilized for training the machine learning models and making predictions.
- **Flask**: Serves as the backend framework for handling web requests.
- **Anvil**: Used for building the frontend web application.
- **Docker & AWS ECS**: For containerizing the application and deploying it in a scalable environment.
- **Web Browser**: Required to access and interact with the web application.

Refer to `requirements.txt` within the `Digit-Classifier` folder to ascertain the specific versions of frameworks and libraries utilized.


<div align="center">
  <img src="https://github.com/ManideepTelukuntla/Digit-Classifier/blob/main/Images/Tech-Stack.png" width="800" height="573" alt="Digit Classifier Tech Stack">
  <br>
  <p>Digit Classifier Tech Stack</p>
</div>

## 5. File Descriptions
- **`Digit-Classifier-Using-CNN-Vs-ViT.ipynb`**: This Jupyter Notebook contains Python code for training, fine-tuning, and visulizating metrics for both CNN & ViT models on MNIST dataset.
- **`Digit-Classifier`**: This folder contain various code files for CNN and ViT models, a Docker configuration file, a Python application file, and a requirements.txt file.
- **`CNN-Assets`**: This folder conatin assets (epoch-accuracy-graph, confusion matrix, misclassifiactions vs correct classifictaions) related to CNN before retraining the model on entire training dataset.
- **`CNN-Assets-Retrained-Model`**: This folder conatin assets (confusion matrix, misclassifiactions vs correct classifictaions) related to CNN after retraining the model on entire training dataset.
- **`ViT-Assets`**: This folder conatin assets (epoch-accuracy-graph, confusion matrix, misclassifiactions vs correct classifictaions) related to ViT before retraining the model on entire training dataset.
- **`ViT-Assets-Retrained-Model`**: This folder conatin assets (confusion matrix, misclassifiactions vs correct classifictaions) related to ViT after retraining the model on entire training dataset.
- **`Images`**: Folder with images pertaining to this project.

## 6. Data Pre-processing
The MNIST dataset undergoes normalization and reshaping to prepare for CNN model training, ensuring pixel values are scaled appropriately. For ViT, images are segmented into patches with added positional encodings to maintain spatial relationships.

## 7. Model Implementation
### CNN Architecture
- **Accuracy Achievements**: The CNN model reached a validation accuracy of 99.07% and, upon retraining, showed 99.77% accuracy on the training set and 98.97% accuracy on the test set.
- **Design**: Comprises layers tailored for feature extraction and classification, including convolutional layers with ReLU activation, max pooling for dimensionality reduction, and dense layers for final classification.
- **Training and Optimization**: Initially trained for 25 epochs, subsequent analysis led to retraining for an optimized 14 epochs, balancing performance and computational efficiency.

### ViT Architecture
- **Accuracy Achievements**: The ViT model demonstrated a validation accuracy of 98.42%. After retraining, it achieved 99.09% training accuracy and 98.69% testing accuracy.
- **Design**: Implements the transformer architecture, treating images as sequences of patches. It incorporates multi-head self-attention mechanisms, layer normalization, and MLPs, concluding with a classification head for digit prediction.
- **Training and Optimization**: Varied epochs and hyperparameter tuning were explored to find the ideal training duration, settling on 43 epochs for retraining to maximize accuracy.

## 8. Analysis and Recommendations
Misclassification analysis revealed patterns and challenges in recognizing certain digits, notably between similar shapes and stylistic variations. This insight led to the following recommendations for model improvement:

### Key Observations
- **Ambiguity in Writing Style**: Misclassifications between digits with similar styles, such as 9 and 4 or 5 and 6, suggest the models' difficulty in distinguishing based on unique handwriting nuances.
- **Overlap in Digit Shapes**: Confusions, notably between 3 and 5, indicate challenges in recognizing subtle shape differences, pointing towards potential improvements in feature extraction and classification capabilities.

### Recommendations
- **Augmenting Training Data**: Incorporating a broader range of handwriting styles and forms in the training data can improve the model's robustness to varied inputs.
- **Enhanced Pre-processing Techniques**: Applying more sophisticated data augmentation methods may help models learn more generalized features, reducing misclassification.
- **Continuous Model Tuning**: Regularly updating the models with new data and feedback can refine their accuracy and adaptability to real-world variations in digit handwriting.

## 9. Conclusion
The Digit Classifier Project showcases the integration of cutting-edge machine learning technologies into a scalable, user-centric application. By addressing the complexities of digit recognition through continuous model evolution and leveraging a cloud-based architecture, this project demonstrates the potential of advanced machine learning models to provide tangible, accessible solutions for digit classification tasks.

## 10. License
Licensed under [MIT License](https://github.com/ManideepTelukuntla/InvestigateTMDBMovieData/blob/master/LICENSE)
