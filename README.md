# ANALYSIS OF POLITICAL TEXTS


This project implements a text classification model to differentiate between texts related to **humanism** and **neoliberalism**. The model is trained using a set of pre-processed text files and outputs predictions on unseen data.

The project utilizes **deep learning techniques** for text classification and is designed for research in the area of **information retrieval** and **document classification**. This project includes all the necessary files for vocabulary extraction, model training, testing, and analysis.

<p align= "center">
    <img src="https://github.com/user-attachments/assets/b2764f60-4257-4b26-9748-4c9ba500c8d7" style="width: 50%; height: auto;">
</p>

## Table of Contents
1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Model Training and Testing](#model-training-and-testing)
4. [File Descriptions](#file-descriptions)
5. [How to Run](#how-to-run)
6. [Requirements](#requirements)

## Project Overview

The project involves the creation of a machine learning model that can classify documents based on whether they are related to **humanism** or **neoliberalism**. The dataset consists of text files from these two categories, which are processed to build vocabularies and train the model.

The final model is saved and can be used for testing new documents. It is based on deep learning architectures and is designed to handle text preprocessing, tokenization, and text classification tasks.

## Directory Structure

The project is organized as follows:

```plaintext
/fold
    /fold/humanismo          # Contains training files related to humanism (.txt files)
    /fold/neoliberalismo      # Contains training files related to neoliberalism (.txt files)
/modelos                     # Directory where the trained models are saved
/pruebas                     # Contains new text files for testing the trained model
corpus.txt                   # List of all unique words in both "humanism" and "neoliberalism" texts
vocabulario_humanismo.txt    # List of unique words from humanism text files
vocabulario_neoliberalismo.txt# List of unique words from neoliberalism text files
PFRY.py                      # Main Python file for training the model and handling classification
```

## Model Training and Testing

### Vocabulary Extraction

The first step in the process is extracting vocabularies from the text files in the `humanismo` and `neoliberalismo` folders. The extracted vocabularies are saved into their respective files (`vocabulario_humanismo.txt`, `vocabulario_neoliberalismo.txt`, and `corpus.txt`).

### Model Training

The model is trained using text data from both categories. The main script (`PFRY.py`) handles text preprocessing, tokenization, and the training process. The trained model is saved in the `/modelos` directory.

### Model Testing

Once trained, the model can classify new documents. Place test files in the `/pruebas` directory, and the model will predict whether each document is related to **humanism** or **neoliberalism**.

## File Descriptions

-   **/fold/humanismo/**: Contains all text files for humanism-related training.
-   **/fold/neoliberalismo/**: Contains all text files for neoliberalism-related training.
-   **/modelos/**: This directory contains the trained model and auxiliary files (e.g., tokenizer and label encoder).
    -   Example model files:
        -   `08710_12epochs_label_encoder.pickle`: Stores label encoder for model interpretation.
        -   `08710_12epochs_modelo.h5`: The trained Keras model file.
        -   `08710_12epochs_tokenizer.pickle`: Tokenizer used for preprocessing the text data.
-   **/pruebas/**: Place new text files here to be classified by the model.
-   **corpus.txt**: Contains all unique words across both categories (humanism and neoliberalism).
-   **vocabulario_humanismo.txt**: Contains all unique words from humanism training files.
-   **vocabulario_neoliberalismo.txt**: Contains all unique words from neoliberalism training files.
-   **PFRY.py**: The main script responsible for training the model and performing text classification. It contains:
    -   Text loading and preprocessing functions.
    -   Model training and saving logic.
    -   Testing functionality for classifying new documents.

## How to Run
### Prerequisites

Ensure you have Python 3.x installed with the following libraries:

```bash
pip install tensorflow keras numpy scikit-learn
```
### Training the Model

1.  Place your training data in the `/fold/humanismo` and `/fold/neoliberalismo` directories.
2.  Run the main Python file to train the model:

```bash
python PFRY.py
``` 

This will generate the trained model and save it in the `/modelos` directory.

### Testing the Model

1.  Place new text files in the `/pruebas` directory.
2.  Run the following command to classify the new files:

```bash
python PFRY.py --test
```

The script will output predictions for each file in the `/pruebas` directory, indicating whether it is classified as **humanism** or **neoliberalism**.

## Requirements

-   Python 3.x
-   TensorFlow
-   Keras
-   Scikit-learn
-   NumPy

### Installing Required Libraries

Run the following command to install all necessary Python libraries:

```bash
pip install -r requirements.txt
```
