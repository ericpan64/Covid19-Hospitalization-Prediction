<<<<<<< HEAD
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
=======
# Model for COVID-19 DREAM Challenge: Question 2

## Overview

Here we describe how to build and run locally an example model provided for Challenge Question 2 of the [COVID-19 DREAM Challenge](https://www.synapse.org/#!Synapse:syn18404605). The goal of this continuous benchmarking project is to develop models that take as input the electronic health records (EHRs) of a patient and outputs the probability of a COVID-19-positive patient to be hospitalized within 21 day after his/her recent COVID-measurement.

## Description of the model

This model takes all features in the data set. First, we generate a feature set for each patient in the `/data` folder using the features and then we apply a 10-fold cross-validation model of choice [Logistic Regression, Support Vector Machine, Random Forest] on the feature set. Once the model is trained, we save the model file in the `/model`folder.

During the inference stage, we create a feature matrix using the same set of demographics features. Then we load the trained model and apply the model on the feature matrix to generate a prediction file as `/output/predictions.csv`



## Dockerize the model

1. Start by cloning this repository.

2. Build the Docker image that will contain the move with the following command:

    ```bash
    docker build -t awesome-covid19-q2-model:v1 .
    ```

## Run the model locally on synthetic EHR data

1. Go to the page of the [synthetic dataset](https://www.synapse.org/#!Synapse:syn21978034) provided by the COVID-19 DREAM Challenge. This page provides useful information about the format and content of the synthetic data.

2. Download the file [synthetic_data.tar.gz](https://www.synapse.org/#!Synapse:syn22043931) to the location of this example folder (only available to registered participants).

3. Extract the content of the archive

    ```bash
    $ tar xvf synthetic_data.tar.gz
    x synthetic_data/
    x synthetic_data/training
    x synthetic_data/evaluation
    ```

4. Create an `output` , `model` and `features` folders.

    ```bash
    mkdir features output model
    ```

5. Run the dockerized model to train on the patients in the training dataset.

    ```bash
    docker run \
        -v $(pwd)/synthetic_data/training:/data:ro \
        -v $(pwd)/output:/output:rw \
        -v $(pwd)/scratch:/scratch:rw \
        -v $(pwd)/model:/model:rw \
        awesome-covid19-q2-model:v1 bash /app/train.sh
    ```

6. Run the trained model on evaluation dataset and generate the prediction file.

    ```bash
    docker run \
        -v $(pwd)/synthetic_data/evaluation:/data:ro \
        -v $(pwd)/output:/output:rw \
        -v $(pwd)/scratch:/scratch:rw \
        -v $(pwd)/model:/model:rw \
        awesome-covid19-q2-model:v1 bash /app/infer.sh
    ```

7. The predictions generated are saved to `/output/predictions.csv`. The column `person_id` includes the ID of the patient and the column `score` the probability for the COVID-19-positive patient to be hospitalized within 21 days after the latest COVID test.

    ```text
    $ cat output/predictions.csv
    person_id,score
    0,0.6153846153846154
    1,0.5384615384615384
    2,0.5384615384615384
    3,0.3076923076923077
    ...
    ```

## Submit this model to the COVID-19 DREAM Challenge

This model meets the requirements (As of Dec 2020) for models to be submitted to Question 2 of the COVID-19 DREAM Challenge. Please see [this page](https://www.synapse.org/#!Synapse:syn21849255/wiki/602419) for updated requirements on how to submit this model.


## Pushing Model to Synapse

In order to push a Docker image to the Docker repository of your Synapse project, the image must be renamed using the following command:

    ```
    docker tag awesome-covid19-q2-model:v1 docker.synapse.org/syn23593381/awesome-covid19-q2-model:v1
    ```

Where ```syn23593381``` must be replaced by the Synapse ID of your project and if desired modify the version number ```v1``` to track your model versions.

In your terminal, login to the Synapse Docker registry using your Synapse credentials:

```docker login docker.synapse.org```

Push the Docker image to your Synapse Project using the following command. Note that you must be a Certified Synapse User in order to be able to push a Docker image to the Synapse Docker registry.

```docker push docker.synapse.org/syn23593381/awesome-covid19-q2-model:v1```


## Submit this model to the COVID-19 DREAM Challenge

This model meets the requirements for models to be submitted to Question 2 of the COVID-19 DREAM Challenge. Please see [this page](https://www.synapse.org/#!Synapse:syn21849255/wiki/602419) for instructions on how to submit this model.

In short 
- Make sure you've created a project like in https://www.synapse.org/#!Profile:3416236/projects. You can access it by clicking on your Profile icon > project, after you've logged in.
- Create an alias of your docker image that you've built. `docker tag awesome-covid19-q2-model:v1 docker.synapse.org/syn23593381/example_baseline`. the `syn23593381` must match your project ID. If you do `docker images` you'll see that both have the same image IDs.
- Login to Synapse's docker hub: `docker login docker.synapse.org` and use your Synapse login credential
- Push up the image `docker push docker.synapse.org/syn23593381/example_baseline`
- Now go back to your project page and click on the "Docker" tab. You should see the image there.
- Click on the image > Click on "Docker repository tool" on the right hand side. > Click "Submit docker repository to a challenge" in the dropdown.
- You'll get an email once the result is back. 
 

## What happened after submission

The two docker commands you ran earlier will be run on the DREAM platform, this time with the real EHR data instead of synthetic data. You will get an email then to 
>>>>>>> 8b57cb2f6f6a8c8aeea8165632fe28b667ad534b
