# Knowledge Source Integration on Medical Code Prediction

This repository is the official implementation of [Knowledge Source Integration on Medical Code Prediction](https://dl.acm.org/doi/10.1145/3308558.3313485).

>📋  The article titled "Improving Medical Code Prediction from Clinical Text via Incorporating Online Knowledge Sources" discusses the importance of translating clinical notes into a standardized format that can be read by computers. This process is typically performed by trained staff and is both time-consuming and expensive. To address this problem, there is a growing interest in developing automatic code assignment models. However, it remains a challenge to fully automate the process. This article proposes a solution to address this challenge. This paper discussed the end-to-end framework called Knowledge Source Integration (KSI), which enables the incorporation of external knowledge while training a model. 


## Requirements

1. Install Python3, enable Cuda on the machine.  <br>
2. Install packages using pip. The following packages needs: stop-words, numpy, scikit-learn. <br>
```setup
pip install numpy scikit-learn stop-words
```
3. Download the two datasets needed for this code: NOTEEVENTS.csv and DIAGNOSES_ICD.csv. These datasets can be downloaded from the MIMIC-III database.


## Data Processing

1. Download the two CSV files, "NOTEEVENTS.csv" and "DIAGNOSES_ICD.csv" <br>
```setup
  You can download two CSV files here:
- [NOTEEVENTS.csv](https://drive.google.com/file/d/13fs4Zn-LyOtqBHgp0V9FN8PQi3LIMN53/view?usp=sharing).
- [DIAGNOSES_ICD.csv](https://drive.google.com/file/d/1VG51aodS4omPDcIv6m2oQFnjxZwRawmc/view?usp=sharing).
```
2. Place the two CSV files in the same directory as the following files and directories: IDlist.npy, wikipedia_knowledge, p1, p2 and p3. <br>
3. Run the following commands in order to preprocess the data: 
``` setup
python3 p1.py, python3 p2.py, and python3 p3.py  
```
4. Once these scripts have completed running, the fully preprocessed dataset ready for use.  <br>

## Training

1.To train the CNN and KSI+CNN models in the paper, run this command:
```train
python KSI_CNN_RZ_20230506.py
```
2.To train the RNN and KSI+RNN (LSTM) models in the paper, run this command:
```train
KSI_LSTM_RZ_20230507.py
```
3.The evaluation metrics will show up automatically after the model run

## Results

Our model achieves the following performance on :

| Evaluation    | CNN Alone | CNN + KSI | LSTM Alone| LSTM + KSI|
| ------------- | --------- | --------- | --------- | --------- |
| Test loss     |  0.0387   |  0.0366   |  0.0341   |  0.0318   |
| Top 10 acc    |  0.7553   |  0.7801   |  0.7617   |  0.7901   |
| Macro AUC     |  0.8323   |  0.8659   |  0.8360   |  0.8710   |
| Micro AUC     |  0.9675   |  0.9737   |  0.9687   |  0.9758   |
| Macro F1      |  0.2199   |  0.2447   |  0.1996   |  0.2471   |
| Micro F1      |  0.6289   |  0.6445   |  0.6423   |  0.6552   |

## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
