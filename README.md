---
title: Getaround price predictor
short_description: Train rental price prediction model and log experiment to MLFlow
---
![MLflow Logo](https://mlflow.org/images/MLflow-logo.png)

# Pour enregistrer un modèle d'apprentissage dans un espace Huggingface Mlflow frdepuis un container docker

- 	répertoire docker
	    - docker build . -t getaround_rental_price_predictor
	    - cd ..
	    - docker run -it --env-file secrets.env -v "$(pwd):/home/app" -e PORT=7860 getaround_rental_price_predictor
- secrets.env
    - MLFLOW_EXPERIMENT_NAME
    - MLFLOW_TRACKING_URI