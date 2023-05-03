>📋  

# Knowledge Source Integration on Medical Code Prediction

This repository is the official implementation of [Knowledge Source Integration on Medical Code Prediction](https://dl.acm.org/doi/10.1145/3308558.3313485).

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install numpy scikit-learn stop-words
```

>📋  1. Install Python3  2. Install packages using pip. The following packages needs: stop-words, numpy, scikit-learn. 3. Download the two datasets needed for this code: NOTEEVENTS.csv and DIAGNOSES_ICD.csv. These datasets can be downloaded from the MIMIC-III database.


## Data Processing

You can download two CSV files here:
- [NOTEEVENTS.csv](https://drive.google.com/file/d/13fs4Zn-LyOtqBHgp0V9FN8PQi3LIMN53/view?usp=sharing).
- [DIAGNOSES_ICD.csv](https://drive.google.com/file/d/1VG51aodS4omPDcIv6m2oQFnjxZwRawmc/view?usp=sharing).

>📋  1. Download the two CSV files, "NOTEEVENTS.csv" and "DIAGNOSES_ICD.csv"
     2. Place the two CSV files in the same directory as the following files and directories: IDlist.npy,          wikipedia_knowledge, p1, p2 and p3.
     3. Run the following commands in order to preprocess the data: python3 p1, python3 p2, and python3 p3.
     4. Once these scripts have completed running, the fully preprocessed dataset ready for use.

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).


## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
