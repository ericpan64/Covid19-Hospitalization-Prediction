# C19-Hospitalization-Likelihood
Here we present a framework to generate hospitalization likelihood for COVID-positive patients within 21 days of test result. 

This is part of the COVID-19 DREAM challenge (specifically [Question 2](https://www.synapse.org/#!Synapse:syn21849255/wiki/602406)).

## Problem Overview
Input: [OMOP-based](https://github.com/OHDSI/CommonDataModel/wiki) patient dataset

Output: Prediction scores (between 0 and 1) for each patient representing likelihood of hospitalization within 21 days of a positive COVID test

Motivation: Accurately predicting patients with high risk for complications can improve medical scheduling and triage for scarse resources

## Links
- [DREAM Challenge Q2 Reference](https://www.synapse.org/#!Synapse:syn21849255/wiki/602406)
- [OMOP Data Model Wiki](https://github.com/OHDSI/CommonDataModel/wiki)

## Description of the model

This model takes all features in the data set. First, we generate a feature set for each patient in the `/data` folder using the features and then we apply a 10-fold cross-validation model of choice [Logistic Regression, Support Vector Machine, Random Forest] on the feature set. Once the model is trained, we save the model file in the `/model`folder.

During the inference stage, we create a feature matrix using the same set of demographics features. Then we load the trained model and apply the model on the feature matrix to generate a prediction file as `/output/predictions.csv`

## Model Usage Details

The instructions to run and submit the model are provided in [SUBMISSION_INSTRUCTIONS.md](SUBMISSION_INSTRUCTIONS.md)