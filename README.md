# Predicting Covid-19 Hospitalization using OMOP Data and Biomedical Text 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Here we (Eric, Mohammad, Santina) present a framework to generate hospitalization likelihood for COVID-positive patients within 21 days of test result. 

This project is part of the COVID-19 DREAM challenge (specifically [Question 2](https://www.synapse.org/#!Synapse:syn21849255/wiki/602406)) and came together while taking Georgia Tech's [OMSCS 6250: Big Data for Health](https://omscs.gatech.edu/cse-8803-special-topics-big-data-for-health-informatics) class. In addition to using the data provided in the DREAM challenge, we incorporate an NLP-based analysis on the separate open-source [CORD-19 dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) and use it to guide feature selection. Overall, incorporating the NLP component without filtering did not improve the results locally, however after filtering we were able to consistetly finish in the top 10 submissions based on AUPR and AUROC (ranking as-of the 11-11-2020 dataset, leaderboard linked [here]((https://www.synapse.org/#!Synapse:syn21849255/wiki/605131)).

## Project Overview
Without a doubt, COVID-19 has dramatically affected all of our lives and it is important we seek to understand and support each other during these challenging times. Our team was motivated by the open-source efforts to better understand COVID-19 and the potential of accurate hospitalization predictions to improve medical scheduling and triage for scarse resources. This repository contains our project submission and analysis.

This project has two key components:
1. Generating a ML model using traditional statistical analysis on the DREAM challenge dataset
2. Performing an NLP analysis on the separate CORD-19 dataset and using that to guide feature selection

In this analysis, features are represented by concept_ids as defined by the OMOP data model from OHDSI. An index of all concept_ids can be found using the OHDSI ATHENA tool [linked here](https://athena.ohdsi.org/).

### Data (from CORD-19 Dataset)

- _Input_: [CORD-19](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) dataset containing research paper abstracts and titles
- _Output_: List of concept_ids based on appearance count in titles and abstracts

### Data (from DREAM Challenge Dataset)

- _Input_: [OMOP-formatted](https://github.com/OHDSI/CommonDataModel/wiki) dataset, results from NLP analysis
- _Output_: Prediction scores (between 0 and 1) for each patient representing likelihood of hospitalization within 21 days of a positive COVID test

## Description of the model

There are two variants we explored during this project: a model trained using all possible concept_ids as features (in src/use_all_concepts), and a model trained using the counts of user-specified concept_ids as features (in src/use_compressed_concepts). For our final submission we used the "compressed" version was easier to deploy already performed significantly well.

For each model, we first generate a feature set for each patient in the `/data` folder using PCA on the counts specified concept_ids, and then we apply a 10-fold cross-validation model of choice [Logistic Regression, Support Vector Machine, Random Forest] on the feature set. Once the model is trained, we save the model file in the `/model`folder. During the inference stage, we create a feature matrix using the same set of demographics features. Then we load the trained model and apply the model on the feature matrix to generate a prediction file as `/output/predictions.csv`

## Run the model locally on synthetic EHR data

1. Download the repository and build the Docker image. The model results will be run on the [synthetic_data](/synthetic_data) folder. For this example, the container name ```awesome-covid19-q2-model``` is arbitrarily used.

    ```bash
    docker build -t awesome-covid19-q2-model:v1 .
    ```

2. Run the dockerized model to train on the patients in the training dataset.

    ```bash
    docker run \
        -v $(pwd)/synthetic_data/training:/data:ro \
        -v $(pwd)/output:/output:rw \
        -v $(pwd)/model:/model:rw \
        awesome-covid19-q2-model:v1 bash /app/train.sh
    ```

3. Run the trained model on evaluation dataset and generate the prediction file.

    ```bash
    docker run \
        -v $(pwd)/synthetic_data/evaluation:/data:ro \
        -v $(pwd)/output:/output:rw \
        -v $(pwd)/model:/model:rw \
        awesome-covid19-q2-model:v1 bash /app/infer.sh
    ```

4. The predictions generated are saved to `/output/predictions.csv`. The column `person_id` includes the ID of the patient and the column `score` the probability for the COVID-19-positive patient to be hospitalized within 21 days after the latest COVID test.

    ```text
    $ cat output/predictions.csv
    person_id,score
    0,0.05468081015726079
    1,0.11703154589067764
    2,0.12081316578916707
    3,0.10190242888652237
    ...
    ```

## Model Submission
The instructions to submit the model are provided in [SUBMISSION_INSTRUCTIONS.md](SUBMISSION_INSTRUCTIONS.md). Feel free to build off-of this as you see fit!