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
        -v $(pwd)/scratch:/scratch:rw \
        -v $(pwd)/model:/model:rw \
        awesome-covid19-q2-model:v1 bash /app/train.sh
    ```

3. Run the trained model on evaluation dataset and generate the prediction file.

    ```bash
    docker run \
        -v $(pwd)/synthetic_data/evaluation:/data:ro \
        -v $(pwd)/output:/output:rw \
        -v $(pwd)/scratch:/scratch:rw \
        -v $(pwd)/model:/model:rw \
        awesome-covid19-q2-model:v1 bash /app/infer.sh
    ```

4. The predictions generated are saved to `/output/predictions.csv`. The column `person_id` includes the ID of the patient and the column `score` the probability for the COVID-19-positive patient to be hospitalized within 21 days after the latest COVID test.

    ```text
    $ cat output/predictions.csv
    person_id,score
    0,0.6153846153846154
    1,0.5384615384615384
    2,0.5384615384615384
    3,0.3076923076923077
    ...
    ```

## Model Submission
The instructions to submit the model are provided in [SUBMISSION_INSTRUCTIONS.md](SUBMISSION_INSTRUCTIONS.md)