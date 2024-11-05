# Bain ML Prediction Pipeline Challenge

Machine Learning prediction pipeline to be implemented on a provided real estate dataset to predict housing price of properties in Chile. The training script was also provided as a Jupyter Notebook.

## Instructions

**Training pipeline**:
First, the model needs to be trained. Use the following `docker compose` command.
> It is assumed that you have access to the train (train.csv) and test (test.csv) dataset and that they are placed inside `model/data/` 
```
docker compose up --build train
```
**Serving**
When the training is completed, just run
```
docker compose up --build serve
```
After which the API's documentation and endpoints can be accessed via **localhost:8000/docs**
> The API keys needed to effectively use the endpoint are located on `db.py`

## Dependencies
```
category-encoders==2.6.4
scikit-learn==1.5.2
numpy==2.1.3
pandas==2.2.3
fastapi==0.115.4
uvicorn==0.32.0
```
> The dependencies are place inside a `requirements.txt` at the root project's directory

## Assumptions

1. For this challenge, and as stated in its instructions, the model quality already satisfied the client's initial needs, so any kind of optimization of the model itself was not performed;
2. For now, too be simple, only the model training metrics and API's payloads/predictions are to be logged
 3. The user has access to the training and testing data to replicate the training pipeline.

## Improvements

Several improvements can be added to this project:
- Suite of tests, both unit and integration, to ensure code quality and reliability;
- Improved security of the API, which the users defined on `db.py` script should be place inside a secure key management service, such as AWS' SecretsManager and accessed through AWS' Python SDK;
- Going forward, model optimization will be key to gain a competitive edge and ensure the business' success, so a more robust Python class that handles model experimentation with various kinds of model and techniques can be built (alternatively, a framework such as MLflow can be used);
- Flesh out the model's training and API's payloads/predictions logging, as to record more information about the model's operation as a whole.

## Database Connection Abstraction
For a simple abstraction of the database connection, AWS' services were used to demonstrate how the client can store his/her tabular files on a S3 Bucket, which in turn can be queried using Athena. This service can then be used to consult the desired database using Python AWS' SDK.

https://lucid.app/documents/embedded/1a61c9c8-0c80-4b2e-b353-5ce38cf85b1f
