
## Dockerize the model

For this example, the container name ```awesome-covid19-q2-model``` is is arbitrarily used.

Build the Docker image that will contain the move with the following command: 

    ```
    docker build -t awesome-covid19-q2-model:v1 .
    ```

## Submit this model to the COVID-19 DREAM Challenge

This model meets the requirements (as of Dec 2020) for models to be submitted to Question 2 of the COVID-19 DREAM Challenge. Please see [this page](https://www.synapse.org/#!Synapse:syn21849255/wiki/602419) for updated requirements on how to submit this model.


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