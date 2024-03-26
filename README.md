
# MGATCDA

----
This repository contains the PyTorch implementation of *Predicting CircRNA-Mediated Drug Sensitivity through Multi-head Graph Attention Networks*. 

**Update on 03/26/2024** 

## Overview

Discovering associations between circular RNAs (circRNAs) and cellular drug sensitivity is essential for understanding drug efficacy and therapeutic resistance. 
Traditional experimental methods to verify such associations are costly and timeconsuming. Thus, the development of efficient computational methods for predicting circRNA-drug associations is crucial. 

In this study, we introduce a novel computational predictor called MGATCDA, aimed at predicting potential circRNA-drug sensitivity associations.

![modeloverview.png](https://github.com/ZiyuFanCSU/MGATCDA/blob/main/img/modeloverview.png)

## Requirements
- python==3.8
- pytorch==2.0.1
- pytorch-cuda==11.8
- rdkit==2022.09.1
- dgl==1.1.2.cu118
- scikit-learn==1.3.2
- numpy==1.23.5
- pandas==1.5.2
- tqdm==4.65.0
- torch-cluster==1.6.3+pt20cu118
- torch-geometric==2.4.0
- torch-scatter==2.1.2+pt20cu118   
- torch-sparse==0.6.18+pt20cu118 
- torch-spline-conv==1.2.2+pt20cu118 

## Data
- Original data source：

The datasets were derived from sources in the public domain:

The circRNA-drug sensitivity associations from https://hanlab.tamhsc.edu/cRic/.

The sequences of host genes of circRNAs from: https://www.ncbi.nlm.nih.gov/gene.

The structure data of drugs from https://pubchem.ncbi.nlm.nih.gov/. 

The names of the drugs and circRNAs used in the experiment can be found on: https://github.com/yjslzx/GATECDA.

The SMILES of drugs can be found in: ./data/data_smile.npy.

- Processed feature data：

5-fold cross-validation data in: ./five_ten_cv_data/five/

10-fold cross-validation data in: ./five_ten_cv_data/ten/

## Train
The training process with default parameters requires a GPU card or CPU.

Run `train.py` using the following command:
```bash
nohup python -u train.py --lr 0.00005 --data_path './five_ten_cv_data/five' --model_dir "./trained_model" >> ./log.log 2>&1 &
```
If you want to perform ten-fold cross-validation, change "--data_path './five_ten_cv_data/five'" to "--data_path './five_ten_cv_data/ten'"

## Contact
If you have any questions, please contact [fzy_china@csu.edu.cn].
