# MLPipeline_Template

### Table of Content
<details><summary>Expand/Collapse</summary>
<br>

Project Setup Steps
- [Create GitHub Repository](#create-github-repository)
- [Clone Github Repository](#clone-github-repository)
- [Create Folder Directory](#create-folder-directory)
- [Create Environment](#create-environment)
- [Setup.py](#setuppy)
- [Edit Requirements.txt](#edit-requirementstxt)
- [Test Logger and Exception in Main.py](#test-logger-and-exception-in-mainpy)
- [Edit Common.py](#edit-commonpy)

Workflow Setup
- [Constants](#constants)
- [Update config.yaml](#update-configyaml)
- [Update schema.yaml](#update-schemayaml)
- [Update params.yaml](#update-paramsyaml)
- [Update the entity](#update-the-entity)
- [Update the configuration manager in src config](#update-the-configuration-manager-in-src-config)
- [Update the components](#update-the-components)
- [Update the pipeline ](#update-the-pipeline)
- [Update the main.py](#update-the-mainpy)
- [Update the app.py](#update-the-apppy)

</details>

# Project Setup Steps

## Create GitHub Repository

On Github Website
1. Create new repository
2. Copy repository code

## Clone Github Repository

In GitHub Folder Bash

Command to clone github repository
```
git clone http://
```

To change to project folder directory

```
cd Project_Folder_Name
```

To open visual studio code application
```
code .
```

## Create Folder Directory
(In Visual Studio Code)

1. Create **template.py**
2. Run **template.py** in cmd terminal

```
python template.py
```
3. Edit Project Name and add folders as needed to list

```
project_name = ""
```

4. push change to github in cmd terminal
```
git add .

git commit -m "update message"

git push -u origin main
```

## Create Environment
In command terminal

```
conda create -n myenv python=3.8 -y

conda activate myenv
```



## Setup.py
[setup.py](setup.py) is a module used to build and distribute Python packages. It typically contains information about the package, such as its name, version, and dependencies, as well as instructions for building and installing the package. 

This information is used by the **pip tool**, which is a package manager for Python that allows users to install and manage Python packages from the command line. By running the setup.py file with the pip tool, you can build and distribute your Python package so that others can use it.


```
REPO_NAME = ""  # Github repository name
AUTHOR_USER_NAME = "aprilhong"
SRC_REPO = ""   # project_name in template.py
AUTHOR_EMAIL = "aprilhong62@gmail.com"
```

## Edit Requirements.txt


1. In [requirements.txt](requirements.txt) add required packages
2. "-e ." used to install setup.py when running this requirements.txt
3. Install requirements.txt in terminal

```
pip install -r requirements.txt
```

## Test Logger and Exception in Main.py

Test run 
- [logger.py](src\Proj\logger.py) 
- and [exception.py](src\Proj\exception.py) 

by running [main.py](main.py) in the command terminal. 


```
python main.py
```

Check that terminal returns logger message and log folder/file is created with the same message


## Edit Common.py

[src\Proj\utils\common.py](src\Proj\utils\common.py)

File stores commonly used functions throughout project 


#  Workflow Setup

## Constants

In [src\Proj\constants\__init__.py](src\Proj\constants\__init__.py), ensure the config and params yaml file are set.

```
from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")
```

## Update config.yaml

In [config\config.yaml](config\config.yaml)

For Data Ingestion: edit **source_URL**
```
data_ingestion:
  root_dir: artifacts/data_ingestion
  source_URL: https://github.com/entbappy/Branching-tutorial/raw/master/Chicken-fecal-images.zip
  local_data_file: artifacts/data_ingestion/data.zip
  unzip_dir: artifacts/data_ingestion
```

For Data Validation And Transformation: Edit 
- unzip_data_dir:
- data_path:
```
data_validation:
  root_dir: artifacts/data_validation
  unzip_data_dir: artifacts/data_ingestion/winequality-red.csv
  STATUS_FILE: artifacts/data_validation/status.txt


data_transformation:
  root_dir: artifacts/data_transformation
  data_path: artifacts/data_ingestion/winequality-red.csv

```




## Update schema.yaml

After Data Ingestion
- Obtain column name and datatypes from dataset and update [schema.yaml](schema.yaml)


## Update params.yaml

Update [params.yaml](params.yaml) based on model metrics

## Update the entity

In [src\Proj\entity\config_entity.py](src\Proj\entity\config_entity.py)


- check if any updates to Class Names are required
- Update **parameters** for ModelTrainerConfig
- Update **metrics** for ModelEvaluationConfig

## Update the configuration manager in src config

In [src\Proj\config\configuration.py](src\Proj\config\configuration.py)

check if any updates are required for 
- entities
- functions


For model_trainer_config, 
- update parameters 

For model_evaluation_config
- update metric file
- update mlflow_url

## Update the components
- [data_ingestion.py](src\Proj\components\data_ingestion.py)

- [data_transformation.py](src\Proj\components\data_transformation.py)
  - add additional transformation functions as needed


- [model_evaluation.py](src\Proj\components\model_evaluation.py)
  - Update eval_metrics, log_into_mlflow


- [model_trainer.py](src\Proj\components\model_trainer.py)
  - update model initatier (linear regression/decision tree/etc.)

## Update the pipeline 

- [stage_01_data_ingestion.py](src\Proj\pipeline\stage_01_data_ingestion.py)
- [stage_02_data_validation.p](src\Proj\pipeline\stage_02_data_validation.py)
- [stage_03_data_transformation.py](src\Proj\pipeline\stage_03_data_transformation.py)
- [stage_04_model_trainer.py](src\Proj\pipeline\stage_04_model_trainer.py)
- [stage_05_model_evaluation.py](src\Proj\pipeline\stage_05_model_evaluation.py)




## Update the main.py



## Update the app.py

```bash
# Finally run the following command
python app.py
```



# Miscellaneous




## MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)


##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/aprilhong/mlproject2.mlflow

MLFLOW_TRACKING_USERNAME=aprilhong 
MLFLOW_TRACKING_PASSWORD=1da17a06a8677064ab52ba22ca059fb39c1d1402

python script.py



Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/aprilhong/mlproject2.mlflow

export MLFLOW_TRACKING_USERNAME=aprilhong 

export MLFLOW_TRACKING_PASSWORD=1da17a06a8677064ab52ba22ca059fb39c1d1402

```



# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 566373416292.dkr.ecr.ap-south-1.amazonaws.com/mlproj

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app




## About MLflow 
MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & tagging your model


