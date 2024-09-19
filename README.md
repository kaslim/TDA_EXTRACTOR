# TDA_EXTRACTOR
# Heart Sound Classification with DTHF Features and CNN

This project is focused on classifying heart sound recordings using features extracted through DTHF (Duration Time Homology Features) and a CNN (Convolutional Neural Network) model. 

## Datasets

Please download the 2016 PhysioNet/Computing in Cardiology Challenge dataset from the following link:

https://archive.physionet.org/challenge/2016/

Once downloaded, extract the dataset and place the files in the dataset directory within this project. The directory structure should look like this:

```bash
project_root/
│
├── dataset/
│   ├── training-a/
│   ├── training-b/
│   ├── training-c/
│   ├── training-d/
│   ├── training-e/
│   ├── training-f/
│   ├── validation/
│   └── REFERENCE.csv
├── extract_DTHF.py
├── main.py
└── ...
```

## Feature Extraction

To extract DTHF features from the dataset, run the extract_DTHF.py script. This will process the heart sound recordings and generate feature files for training and validation:

```bash
python extract_DTHF.py
```
This script will generate .npy files containing the extracted features, which will be saved in the same directory as the dataset.


## Training the Model

After the features have been extracted, you can train the CNN model using the extracted DTHF features. Simply run the main.py script:

```bash
python main.py
```
This will train a CNN model for classifying heart sound recordings as either normal or abnormal, and the trained model will be saved in the project directory.

