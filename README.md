# HeadacheDSS

This repository contains all the code required to reproduce all knowledge incorporation experiments given in the paper: 
'A decision support system to follow up and diagnose chronic primary headache patients using semantically enriched data'. 

It consists of a larger `SemanticProcessor` module and two scripts (`oversampling.py` and `feature_extraction.py`) 
to generate predictions for different over-sampling techniques and different feature extraction techniques. The 
predictions are stored in the output folder, which can then be processed in order to calculate different metrics 
(such as accuracy and Cohen's Kappa score) followed by bootstrap testing in order to test statistical significance.

## 1. Installing all dependencies

## 2. Generating the Knowledge Base with the SemanticProcessor

## 3. Over-sampling with prior knowledge vs ADASYN, SMOTE and weighted samples

## 4. Generating unsupervised features with the Weisfeiler-Lehman kernel
