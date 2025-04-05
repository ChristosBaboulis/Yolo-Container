# YOLO-based Vehicle Speed Analysis in Azure Container App

This project provides a Dockerized application for vehicle detection, tracking, and speed calculation in uploaded videos using YOLOv8 (Ultralytics) and OpenCV. The container is deployed as an Azure Container App and listens for messages from Azure Event Hub (Kafka-compatible), which are triggered by Azure Event Grid upon new video uploads to a Storage Blob Container.

## Architecture Overview

1. **Azure Storage (Blob Container)**  
   - Stores incoming vehicle footage (e.g. dashcam, highway surveillance).  
   - Triggers an Event Grid event upon new video upload.

2. **Azure Event Grid**  
   - Routes the upload event to an Azure Event Hub (Kafka-compatible endpoint).

3. **Azure Event Hub**  
   - Forwards the event (video path and metadata) to subscribed consumers.

4. **Azure Container App (this project)**  
   - Acts as a Kafka consumer.  
   - Downloads the video from blob storage.  
   - Applies YOLOv8 for vehicle detection and OpenCV for tracking and speed analysis.  
   - Logs analytics and violations (e.g. overspeeding vehicles).  
   - Uploads structured logs to Azure Blob Storage.

## Features

- Vehicle detection using YOLOv8 (Ultralytics)  
- Real-time tracking and speed estimation with OpenCV  
- Logs violations and average speed per lane/direction  
- Reads video events from Azure Event Hub (Kafka)  
- Scales automatically via Azure Container Apps  
- Uploads structured logs to Azure Storage  

## Technologies Used

- Python 3.10+  
- Ultralytics YOLOv8  
- OpenCV  
- Azure Event Hub (Kafka API)  
- Azure Blob Storage  
- Azure Event Grid  
- Azure Container Apps  
- Docker  
