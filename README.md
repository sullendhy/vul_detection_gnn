# Vul_detection_gnn
## 1. Environments
- Python 3.8.12 
- transformers 4.24.0 
- torch 1.11
## 2. Datasets
[**CodeXGLUE**](https://arxiv.org/pdf/2102.04664.pdf) datasets used in our work can be found in **dataset** folder. Note we augment original data with code refactoring. Read paper for more details.
- train_aug.jsonl 
- valid.jsonl
- test.jsonl 
## 3. Train models to reproduce our results. 
 
 `python run.py`
 

 We set the hyperparameters to their default values.
## 4. Evaluate model
 `python evaluator.py`

 Trained models are stored in **saved_models** folder.
## 5. Ackonwledgements
Special thanks to [Devign](https://arxiv.org/pdf/1909.03496.pdf) and [ReGVD](https://arxiv.org/pdf/2110.07317.pdf) for their groundbreaking work, upon which this repository is built. Their innovative ideas and code provided the inspiration and starting point for the development of this project.
## 6. Contact us
If you have any questions about this work or paper, please let us know.