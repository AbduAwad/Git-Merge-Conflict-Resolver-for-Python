# VsCode Extension - Git Merge Conflict Resolver for Python (COMP 4107 Final Project)

- **Overview:** This project is a Visual Studio Code extension that automatically resolves merge conflicts in Python files. It uses a Recurrent Neural Network to predict the correct resolution for a given conflict (Branch A / Branch B) as well as a fine-tuned BERT model to generate a custom merge conflict resolution when needed.

_____

### Group Number: 89
_______

### Contributors:

| Name              | Student Number| Email |
| ---               | ---           | --- |
Abdulrahman Awad    | 101256090     | abdulrahmanawad@cmail.carleton.ca
Igor Tascu          | 101181093     | igortascu@cmail.carleton.ca
Antony Ren          | 101151779     | antonyren@cmail.carleton.ca

_______

### Dataset:

Victor Cacciari Miraldo. (2020). Dataset of merge conflicts collected from GitHub repositories [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3751038

Github Mining for Merge Conflict Resolution

# Virtual Machine:

1. Create a new virtual environment using the following command:

```bash
python -m venv myenv
```

2. Activate the virtual environment using the following command: (Windows)
```bash
$ source myenv/Scripts/activate
```

3. Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

_____

### Training the Transformer Model: Transformer.py: 

- Run on google colab using the T4 GPU:

#### Instructions: (works in my google colab environment) just obtain the model from the environment and put it in the models folder in the project when we make it.

- **Transformer.py:** Script to train and output trained bert model on the dataset of merge conflicts on 10 examples in the dataset conflicts-py folder. Works in google colab and takes 10 minutes on GPU. The model is saved in the models folder in google colab. Must enter in API Key from wandb.ai during runtime to make it work.

- To run in google colab, Mount the google drive and run the script. and save the dataset in the google drive. So the model is saved in the models folder in google colab. Then download the model and put it in the models folder in the project.

**Current Approximate Training Time on Entire Set on the T4 GPU: 10.75 hours**

#### Test the Transfomer model 
- run 'test_ransformer.py' in google colab in same notebook but in a different cell under the training cell. 

- **test_transformer.py:** Script to test the trained model by accessing the outputed model in the model_output/final_model folder. You can call the model on a conflict and it will output the predicted resolution.

_____
### Test VsCode Extension:

1. Compile the typescript files using the following command:

```bash
npx tsc
```

2. Run a live instance of the extension in the extension development host:

    - Open the src/extension.ts file in Visual Studio Code
    - press 'ctrl + shift + p' and select 'Debug: Start Debugging'
    - Select 'Extension Development Host' from Dropdown Menu.