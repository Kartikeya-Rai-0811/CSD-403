# Predictive Analysis Project

## Project Overview

This project focuses on building a machine learning system for predictive analysis. The goal is to develop a reproducible end-to-end machine learning project with a clean project structure, data processing, model training, and prediction pipelines.

## Dataset

The project will use the dataset provided for the predictive analysis problem. The raw dataset will be stored inside:

`data/raw/`

Further details about the dataset, features, preprocessing, and modelling will be added as the project progresses through the upcoming units.

## Project Structure

```text
CSD 403/
├── data/
│   └── raw/
├── notebooks/
│   └── 01_eda.py
├── src/
│   ├── __init__.py
│   ├── logger.py
│   ├── exception.py
│   ├── data_prep.py
│   ├── model.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   └── pipeline/
│       ├── train_pipeline.py
│       └── predict_pipeline.py
├── artifacts/
├── requirements.txt
├── README.md
└── .gitignore
