# C19-Hospitalization-Likelihood
Here we present a framework to generate hospitalization likelihood for COVID-positive patients within 21 days of test result. 

This is part of the COVID-19 DREAM challenge (specifically [Question 2](https://www.synapse.org/#!Synapse:syn21849255/wiki/602406)).

## Problem Overview
Input: [OMOP-based](https://github.com/OHDSI/CommonDataModel/wiki) patient dataset
Output: Prediction scores (between 0 and 1) for each patient representing likelihood of hospitalization within 21 days of a positive COVID test
Motivation: Accurately predicting patients with high risk for complications can improve medical scheduling and triage for scarse resources

## Framework Overview
This project uses PySpark for performing feature engineering on the datasets and then applies Logistic Regression to generate the final score. A separate NLP-analysis is used in evaluating novel features (outlined in [Feature Engineering](#FeatureEngineering) section).

### Dataset Analysis
- Overview of available columns, getting familiar with the data

### Feature Engineering
- Technical formatting into features
- Feature filtering for independence
- NLP Sub-Component: CORD dataset to identify predictive features

### ML Models
- Logistic Regression

### Model Evaluation
- AUPR, AUROC, balanced accuracy

## Links
- [DREAM Challenge Q2 Reference](https://www.synapse.org/#!Synapse:syn21849255/wiki/602406)
- [OMOP Data Model Wiki](https://github.com/OHDSI/CommonDataModel/wiki)
