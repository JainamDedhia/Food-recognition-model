Food Recognition Model using ResNet50

This repository contains code for a food recognition model developed using TensorFlow/Keras and ResNet50 architecture. The model identifies various food items from the Food-101 dataset, categorized into a subset of 11 classes.

Features:

Dataset: Utilizes the Food-101 dataset for training, automatically downloaded by the script.
Model Training: Transfer learning with ResNet50 for feature extraction.
Data Handling: Preprocessing with image augmentation and dataset splitting.
Metrics: Evaluation of accuracy and loss during training.
Predictions: Demonstrates classification of custom food images.
Usage:

Clone the repository:
bash
Copy code
git clone https://github.com/JainamDedhia/Prasunet_AD_01.git
cd Prasunet_AD_01

Copy code
python index.py
Execute the Jupyter notebook food_recognition_model.ipynb to train the model and make predictions.
File Description:

food_recognition_model.ipynb: Jupyter notebook containing the complete code for training the food recognition model and making predictions.
download_dataset.py: Python script to download and prepare the Food-101 dataset automatically.
Directory Structure:

images/: Directory where the Food-101 dataset images are stored after downloading.
models/: Directory for saving trained models.
results/: Directory for storing model training logs and evaluation results.
Feel free to explore and modify the code according to your requirements. Contributions and suggestions are welcome!
