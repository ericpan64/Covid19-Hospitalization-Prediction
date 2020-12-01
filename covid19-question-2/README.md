# Example model for COVID-19 DREAM Challenge: Question 2

## Overview

Here we describe how to build and run locally an example model provided for Challenge Question 2 of the [COVID-19 DREAM Challenge](https://www.synapse.org/#!Synapse:syn18404605). The goal of this continuous benchmarking project is to develop models that take as input the electronic health records (EHRs) of a patient and outputs the probability of a COVID-19-positive patient to be hospitalized within 21 day after his/her recent COVID-measurement.

## Description of the model

This example model takes three basic demographics features include age, gender and race. First, we generate a feature set for each patient in the `/data` folder using the demographics feature and then we apply a 10-fold cross-validation logistic regression model on the feature set. Once the model is trained, we save the model file in the `/model`folder.

During the inference stage, we create a feature matrix using the same set of demographics features. Then we load the trained model and apply the model on the feature matrix to generate a prediction file as `/output/predictions.csv`

## Dockerize the model

1. Move to this example folder

2. Build the Docker image that will contain the move with the following command:

    ```bash
    docker build -t awesome-covid19-q2-model:v1 .
    ```
    **You can give whatever name you want the image to be. It's for you to identify it locally** 

    This builds the image based on the Dockerfile, which contains comand copying your train.sh and infer.sh into its image, as well as installing some necessary Python packages. 
    You can think of this as setting up the environment.

## Run the model locally on synthetic EHR data

1. Go to the page of the [synthetic dataset](https://www.synapse.org/#!Synapse:syn21978034) provided by the COVID-19 DREAM Challenge. This page provides useful information about the format and content of the synthetic data.

2. Download the file [q2_synthetic_data_08-19-2020.tar.gz](https://www.synapse.org/#!Synapse:syn22043931) to the location of this example folder (only available to registered participants).

3. Extract the content of the archive

    ```bash
    $ tar xvf q2_synthetic_data_08-19-2020.tar.gz
    x q2_synthetic_data_08-19-2020/
    x q2_synthetic_data_08-19-2020/training
    x q2_synthetic_data_08-19-2020/evaluation
    ```

4. Create an `output` , `model` and any optional folders you need for your scripts

    ```bash
    mkdir output model
    ```

5. Rename the folder to `synthetic_data` and run the dockerized model to train on the patients in the training dataset.

    ```bash
    docker run \
        -v $(pwd)/synthetic_data/training:/data:ro \
        -v $(pwd)/output:/output:rw \
        -v $(pwd)/scratch:/scratch:rw \
        -v $(pwd)/model:/model:rw \
        awesome-covid19-q2-model:v1 bash /app/train.sh
    ```

    This add your local volumes to the docker volumes so that train.sh can access it, which reads from the docker's directory /app for the data to train on.

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

8. You're not required to run through steps 1-7 in order to submit to the competition. This just to help you understand how it works and make sure your code runs as expected.

## Submit this model to the COVID-19 DREAM Challenge

This model meets the requirements for models to be submitted to Question 2 of the COVID-19 DREAM Challenge. Please see [this page](https://www.synapse.org/#!Synapse:syn21849255/wiki/602419) for instructions on how to submit this model.

In short 
- Make sure you've created a project like in https://www.synapse.org/#!Profile:3416236/projects. You can access it by clicking on your Profile icon > project, after you've logged in.
- Create an alias of your docker image that you've built. `docker tag awesome-covid19-q2-model:v1 docker.synapse.org/syn23585053/example_baseline`. the `syn23585053` must match your project ID. If you do `docker images` you'll see that both have the same image IDs.
- Login to Synapse's docker hub: `docker login docker.synapse.org` and use your Synapse login credential
- Push up the image `docker push docker.synapse.org/syn23585053/example_baseline`
- Now go back to your project page and click on the "Docker" tab. You should see the image there.
- Click on the image > Click on "Docker repository tool" on the right hand side. > Click "Submit docker repository to a challenge" in the dropdown.
- You'll get an email once the result is back. Or/and you can check the dashboard here: https://www.synapse.org/#!Synapse:syn21849255/wiki/605229 
 

## What happened after submission

The two docker commands you ran earlier will be run on the DREAM platform, this time with the real EHR data instead of synthetic data. You will get an email then to 