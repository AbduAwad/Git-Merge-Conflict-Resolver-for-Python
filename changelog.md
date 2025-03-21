1. Transformer.py: Trains the BERT model on the dataset of merge conflicts on 10 examples in the dataset conflicts-py folder. Works in google colab and takes 10 minutes on GPU. The model is saved in the models folder in google colab. Must enter in API Key from wandb.ai during runtime to make it work.

2. Mar20 6pm: add RNN classifer first pass, separate data preprocessing funcs into dataPreprocessing.py RNN is buggy will fix later tn

2. Mar21: Complete RNN implementation, gets results -> need to verify accuracies and results further to see what needs to be adjusted