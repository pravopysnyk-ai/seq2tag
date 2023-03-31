# Sequence to Tag Model Training & Evaluation

This repository provides code for training and evaluating Sequence to Tag models for grammatical error correction for the Ukrainian language.

It is mainly based on `PyTorch` and `transformers`.

## Requirements

Our language models were trained in a CUDA-enabled environment. To set everything up, follow the [instructions](https://docs.nvidia.com/cuda/) from the official NVIDIA website.

## Installation

First, you need to initialize the helper submodule directory:

`git submodule update --init`

Then, run the following command to install all necessary packages:

`pip install -r requirements.txt`

The project was tested using Python 3.7.

## Training

To train the models, first initialize the trainer with specified paths to the input data and the model name.

`trainer = Seq2TagTrainer(your_source_file, your_target_file, your_model_name, your_model_path)`

Then, run the training function.

`trainer.train()`

## Inference

To use the inference, initialize the predict and run it:

`predict = Seq2TagPredict`

`predict.main()`

After quick initialization, the program will print the welcome message and ask for input. 
Expected input is supposed to be a full sentence in the Ukrainian language, so the program might handle invalid input incorrectly. 
To exit the program, simply press Enter with no input.
