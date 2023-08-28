# MLflow Custom Container Deployment in Minikube

This repository contains instructions and resources to deploy an MLflow UI using a custom Docker container with SQLite as the backend store in a Minikube cluster.

## Prerequisites

1. [Docker](https://docs.docker.com/get-docker/)
2. [Minikube](https://minikube.sigs.k8s.io/docs/start/)
3. [kubectl](https://docs.docker.com/get-docker/)

## Steps to Run

```
prefect orion start
```
1. **Start minikube**
```
    minikube start
```
2. **Install helm**
   ```
    apt-get install helm
```
3. **Deploy mlflow ui**
```
    helm install -n mlflow mlflow-server ./mlflow-charts
```
4. **Install pip & kubectl**
   ```
    1. yum instal pip
    ```

This command should run mlflow UI up and running at port 5001.

# ML model Deployment and serving in Minikube
1. **Build  docker image**
    ```
    Docker build -t my-model .
```
2. **Run docker container**
   ```
    docker run -p 8001:8001 my-model
    ```
3. **Apply manifests**
    ```
    kubectl apply -f mlflow-deployment.yaml
    ```
    kubectl apply -f mlflow-service.yaml
    ```

4. **Predict and serve models**
