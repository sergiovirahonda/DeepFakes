# DeepFakesTrainingContainer
This repository is part of a project which seeks to explain how to build a Deep Fakes model from scratch. This repo contains the files required to train two Autoencoders using a Docker container which is Google AI Platform ready.

To understand the whole project, please check these Kaggle notebooks in the following order:
* [Deepfakes preprocessing](https://www.kaggle.com/sergiovirahonda/deepfakes-preprocessing)
* [Deepfakes model training](https://www.kaggle.com/sergiovirahonda/deepfakes-model-training)
* [Deepfakes face swapping](https://www.kaggle.com/sergiovirahonda/deepfakes-face-swapping)

# Prerequisites
You must have a [Google Cloud Platform](https://cloud.google.com/) account
Select or create [a project on Google Cloud Platform](https://console.cloud.google.com/projectselector2/home/dashboard)
You must check that [billing is enabled](https://cloud.google.com/billing/docs/how-to/modify-project) in your account
You must enable the [AI Platform Training & Prediction, Compute Engine and Container Registry APIs](https://console.cloud.google.com/flows/enableapi?apiid=ml.googleapis.com,compute_component,containerregistry.googleapis.com)
If you want to test this container locally, make sure that you have installed [nvidia-docker](https://github.com/NVIDIA/nvidia-docker#quickstart)

# Step 0: Setting everything up
* Start by installing and initializing the [Cloud SDK](https://cloud.google.com/sdk/docs/quickstart)
* Install [Docker](https://docs.docker.com/engine/install/) on your machine 
* IMPORTANT: If you're using Linux-based OS such as Ubuntu, then add your username to the Docker group, that way yo don't need to run Docker commands with "sudo": sudo usermod -a -G docker ${USER} and then reinstall your system.
* Use GCloud as the credential helper for Docker: gcloud auth configure-docker

# Step 0.1: Creating variables required for the project

* Specify a name for your bucket using the Project ID of the one that you just created:
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
BUCKET_NAME=${PROJECT_ID}-aiplatform

* Define the region where your model will be stored:
REGION=us-central1 (all regions are available [here](https://cloud.google.com/ml-engine/docs/regions))

* Other variables required:

IMAGE_REPO_NAME=tf_df_custom_container
IMAGE_TAG=tf_df_gpu
IMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_REPO_NAME:$IMAGE_TAG

MODEL_DIR=df_model_$(date +%Y%m%d_%H%M%S)
JOB_NAME=df_job_$(date +%Y%m%d_%H%M%S)

# Step 0.2: Creating a bucket to store your model trained
* To create the bucket, run:
```
gsutil mb -l $REGION gs://$BUCKET_NAME
```

# Step 1: Create the custom model
The [model.py](model.py) file contains the Keras models for the autoencoders. Feel free to modify this file and adapt it to your needs.

The [data_utils.py](data_utils.py) file contains the code necessary to process the data and prepare it to be ingested to the training pipeline. In addition, It allows you to save the final model into Google Cloud Storage so you can download it from there once the training has finished.

The [config.yaml](config.yaml) file contains the instances specification where your model will be trained. It contains now a standard GPU tier. If you want to modify it, keep in mind that multi GPU machines require to use mirrored strategy.

Finally, the [task.py](task.py) file handles the model training. It initializes the GPUs if available, manages mirrored strategy if indicated and also is in charge of receiving the arguments passed to the container.

# Step 2: Create the Docker file
Check the [Dockerfile](Dockerfile) to understand how the Docker image is created and how it interacts with GCP.

# Step 3: Build the Docker Image
Issue the command:
```
docker build -f Dockerfile -t $IMAGE_URI ./
```
This step could take some minutes to complete, keep that in mind.


# Step 4: Push the Docker image to Google Cloud Platform
If you have correctly followed the steps suggested before, you must be able to do this. If you don't please check [this](https://cloud.google.com/container-registry/docs/pushing-and-pulling) documentation, might be helpful for you. To push the image, issue the command:
```
docker push $IMAGE_URI
```
This last command will take some minutes to complete.

# Step 5: Submitting and monitoring the training job onto Google AI Platform.
Once the Docker push process has ended, it's time to train the model on AI platform. To do so, issue the command:
```
gcloud ai-platform jobs submit training $JOB_NAME --region $REGION --master-image-uri $IMAGE_URI --config config.yaml -- --epochs=3000 --bucket-name=$BUCKET_NAME
```
If everything goes well, you should get a success message. To monitor the job, issue the command:
```
gcloud ai-platform jobs describe $JOB_NAME
```
# Step 6: Getting your autoencoders trained
As a final step, go to Google Cloud Storage, open the bucket you've created for this job and you'll find there the folders with the model files within it.