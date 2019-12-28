CSE 256 - Final Project - Vishwas Nagesh Moolimani and Rithesh Rampapura Narasimha Murthy

Goal: To predict short and long answer responses to real questions about Wikipedia articles.
Dataset:
 Google's Natural Questions
 Size of training set: 307,373 examples
 Size of test and validation set: 7,842 examples ; Test set used - 350 examples

Model:
 Customized Bert-Joint Model ! F1 Score - 0.7

You should be able to run:
 > python final_run.py

There are five python files in this folder:

- (final_run.py): Invokes a tkinter based GUI where there are default examples which can be run. There is a lso a provision for custom inputs which can be used as mentioned.

- (bert_joint.py): This file contains the model and the invoking fucntions for the model. All the preprocessing and output postprocessing steps are written in this file

- (bert_utils.py, modelling.py, tokenization.py): These files are support files for bert_joint.py
