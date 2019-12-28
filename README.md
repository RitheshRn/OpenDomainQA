## Open Domain Question Answering using customized BERT-Joint model for Google Natural Questions

#### Goal: To predict short and long answer responses to real questions about Wikipedia articles.<br/>
#### Dataset:<br/>
* Google's Natural Questions<br/>
* Size of training set: 307,373 examples<br/>
* Size of test and validation set: 7,842 examples ; Test set used - 350 examples<br/>

#### Model:<br/>
* Customized Bert-Joint Model ! F1 Score - 0.7<br/>

You should be able to run:
 > python final_run.py

There are five python files in this folder:

- (final_run.py): Invokes a tkinter based GUI where there are default examples which can be run. There is a lso a provision for custom inputs which can be used as mentioned.

- (bert_joint.py): This file contains the model and the invoking fucntions for the model. All the preprocessing and output postprocessing steps are written in this file

- (bert_utils.py, modelling.py, tokenization.py): These files are support files for bert_joint.py
