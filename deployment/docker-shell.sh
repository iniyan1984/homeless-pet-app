#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Read the settings file
source ./environment.shared

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .

# Run the container
docker run --rm --name $IMAGE_NAME -ti \
-v /var/run/docker.sock:/var/run/docker.sock \
--mount type=bind,source="$(pwd)",target=/app \
--mount type=bind,source=$(pwd)/../secrets/,target=/secrets \
--mount type=bind,source=$(pwd)/../api-service,target=/api-service \
--mount type=bind,source=$(pwd)/../frontend-react,target=/frontend-react \
-e GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS \
-e GCP_PROJECT=$GCP_PROJECT \
-e GCP_ZONE=$GCP_ZONE $IMAGE_NAME
