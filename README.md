[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/7CHtmVPJ)
# Final Project Code

**Due: Wednesday, April 26 @ 11:59pm**

Find the complete [rubric](https://ufl.instructure.com/courses/469792/assignments/5548186) in the Canvas assignment.

## Code Implementation

Maintain all Python code that you develop for your project in this repository.

You can use _any_ packages that come as default option with Anaconda, TensorFlow or PyTorch. We need to be able to run your implementation on our machines. So, be sure to get approval from us for any special packages! It we cannot run the code, you will lose a significant number of points.

Your final code submission should include:

* **README.md** - follow the template available [here](https://github.com/catiaspsilva/README-template).This file should include information about library requirements and instructions for running your code.

* **Python scripts** - python script/s or Jupyter Notebook/s to implement your experiments and ideas. Code should be running (free of errors or bugs), well documented, clean of clutter (remove anything not needed).
  * If you are training and testing a model, create two files: a training file and a test file.

Submit the URL of this repository to the [Canvas assignment](https://ufl.instructure.com/courses/469792/assignments/5548186).

## About the project

This is an individual project. Project topic is "EEG based Emotion Recognition by Differential Entropy Feature".  In this project, we investigate the use of differential entropy feature in EEG signals for emotion recognition. Specifically, we aim to test the hypothesis that differential entropy in the gamma-band frequency range provides better emotion recognition accuracy compared to other frequency bands.

## Dataset 

The dataset of this project is from the SJTU Emotion EEG Dataset (SEED), which is a collection of EEG datasets provided by the BCMI laboratory from Shanghai Jiao Tong University in China. It is an publicly available dataset and you can download the dataset by sending email to BCMI laboratory.

There are two dataset folders we used for this project. Folder "Extracted feature" contains differential entropy features extracted with 1-second sliding window. It is the dataset used to train our model and do the experiment. You can obtain it by clicking [here](https://drive.google.com/drive/folders/1ILYZtqdDqdVGjhc9EecLrr5cbVwtCWUx?usp=share_link).

Folder "Prepocessed_EEG" contains downsampled, preprocessed and segmented versions of the EEG data in MATLAB (.mat file). It is the dataset used to extract differential entropy and divide the signal into 5 frequency bands. You can obtain this dataset by clicking [here](https://drive.google.com/drive/folders/1SIqQ6ctcBkHXOTNHQuSbcfWKE3Ud4RWP?usp=share_link).

## Code

There are two part of the code. As same as the dataset, one part is for feature extraction and another part is for training model and experiment.

### For training model and experiment

You can find them in reposiroty. I'm going to illustrate what they are in 'File' section.

### For extracting feature

These codes are in matlab code file and are not used to do the emotion recognition, therefore I didn't push them onto the repository. If you want to try them, you can click [here](https://drive.google.com/drive/folders/1OkuMCvLEI-inf2zUJgjlJuoNIGCdEdiK?usp=share_link).

## Files

### Files you need to download
1. [Extracted feature](https://drive.google.com/drive/folders/1ILYZtqdDqdVGjhc9EecLrr5cbVwtCWUx?usp=share_link) --- folder contains differential entropy features, used for training and experiment. Using after integration.
2. [Available data](https://drive.google.com/drive/folders/1t3Jo6M7vVzJvmalQbo6rMqW3l3A_1WyT?usp=share_link) --- folder contains dataset for training directly. 'train_data.npy' and 'test_data.npy' are dataset of all 5 frequency band. 'train_data_split' and 'test_data_split' are the dataset divide to 5 frequency band. 'train_label' and 'test_label' are the labels
3. [trained_model](https://drive.google.com/drive/folders/1X8F_nBTWH8JcdRI-QBybbRMmwJY1KHwv?usp=share_link) --- folder contains trained models generate from training.ipynb. Used for test.

###  Files in repository
1. training.ipynb --- code used for training the datasets
2. test.ipynb --- code used for test the datasets
3. Data_Integration.py --- code to integrate the dataset from extracted feature.
4. model --- folder contains python code of implemetation of each model.
5. evaluate.py --- code of function for evaluate the model performance.
6. KNN_experiment.ipynb --- code of experiment with knn model
7. DNN_experiment.ipynb --- code of experiment with neural network model.
8. environment.yml --- environment of code.
9. README.md ---- this file.

## How to use

Data integration: "Extracted feature" folder contains the unintegrated dataset of subject 1-15. I use Data_Integration.py to integrate the dataset into 6 files and they are all in 'Available data' folder. You can use this folder to do all the experiment. If you don't want to integrate the dataset, use 'Available data' folder directly.

Training: Training code do not need any parameters to input. After running, it generates 5 model files and 3 history files saved in "trained_model" folder and used for testing. 

Test: Use the evaluate.py to evluate each model, provide the accuracy result. You do not need to test after training, model files are existed in trained model folder and you can utilize them in any time.

Experiment: "KNN_experiment.ipynb" and "DNN_experiment.ipynb" are used for verifying the hypothesis and comparing each model. Do not need any parameters to input.


