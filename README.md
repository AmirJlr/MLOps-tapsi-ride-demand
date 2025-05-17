![GitHub License](https://img.shields.io/github/license/aaghamohammadi/tap30-ride-demand-mlops)
![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)
![code style: black](https://img.shields.io/badge/code%20style-black-black)
![Python Version](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/docker-ready-blue?logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/kubernetes-deployed-blue?logo=kubernetes&logoColor=white)
![FastAPI](https://img.shields.io/badge/api-FastAPI-009688?logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-tracking-blue?logo=mlflow&logoColor=white)

# Tapsi Ride Demand Prediction

Tapsi is a ridesharing company located in Iran, recording approximately 200 million travels in 2024.

The company shares a portion of their data for taxi demand in Tehran, which is divided into grid cells (rows and columns). This MLOps project aims to predict taxi demand at different times for specific areas in the city.

## Dataset Description

The city is modeled as an n x m grid. Every cell of the aforementioned grid represents an area of Tehran. The number of requests for t periods of one hour each is given as the input. The input file is formatted as follows:

*   On the first line, you are given an integer `t` denoting the number of one hour periods.
*   On the second line, there are two integers `m`, `n` denoting the dimensions of the grid.
*   Following these two lines, `t` matrices of size `n x m` are presented.
    *   The first matrix represents the number of requests at time 0.
    *   The second matrix represents the number of requests at time 1.
    *   ...
    *   The `i`-th matrix represents the number of requests at time `i-1`.

Note that time 0 is not necessarily 00:00 a.m. for any day, but any two times `i` and `j` (where `i < j`) are exactly `j - i` hours apart. Some entries have been withheld and are denoted by `-1`.

A sample of the input data is given below:

```
2
4 4
0 3 15 10
-1 -1 9 23
-1 0 0 0
0 0 0 12
0 2 12 11
-1 5 -1 20
3 0 0 0
0 2 0 14
```

## MLOps Pipeline Components

- **Data Ingestion**: Connects to Cloud Object Storage, retrieves data, and stores it locally. This component ensures reproducible and versioned data access, a core MLOps principle.
- **Data Processing**: Prepares and transforms the data to be ready for model training. This includes cleaning, feature engineering, and splitting data, all versioned and automated for consistency.
- **Model Training**: Trains a Random Forest model (using `sklearn`) on the processed data. The training process is logged and versioned, allowing for reproducibility and comparison across experiments.
- **Model Evaluation**: Assesses the trained model's performance using various metrics. This step is crucial for ensuring model quality and for making informed decisions about deployment.
- **Model Serving**: Deploys the trained model as an API endpoint for real-time predictions. This component focuses on scalability, reliability, and monitoring of the deployed model.

## Tools and Technologies for MLOps

- **API**: FastAPI is used to create a robust and high-performance web API for serving model predictions, facilitating easy integration with other services.
- **Experiment Tracking**: MLflow is used for meticulously tracking experiments, including parameters, metrics, and model artifacts. This is fundamental for reproducibility, model comparison, and governance in an MLOps lifecycle.
- **Containerization**: Docker is used to containerize the application, ensuring consistency across different environments (development, testing, production) and simplifying deployment.
- **CI/CD**: GitHub Actions automates the building of Docker images, running tests, and pushing images to Docker Hub. This automates the release pipeline, enabling rapid and reliable updates.
- **Deployment & Orchestration**: Kubernetes is used as the deployment platform, providing scalability, resilience, and efficient management of containerized applications in a production environment.
- **Version Control**: Git (via GitHub) is used for versioning code, data, and configurations, which is essential for tracking changes, collaboration, and rollbacks.
- **Monitoring**: (Placeholder for a chosen monitoring tool, e.g., Prometheus, Grafana) - Essential for observing model performance and system health in production, enabling proactive issue detection and maintenance.
