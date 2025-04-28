Vehicle Detection using YOLOv8

Project Overview

This project demonstrates the application of YOLOv8, a state-of-the-art deep learning model, for vehicle detection. The goal is to identify and classify five types of vehicles: Bus, Car, Motorcycle, Pickup, and Truck. The model was trained using a custom dataset and provides accurate predictions based on image inputs.

Key Points:
Dataset: Custom vehicle detection dataset with images in the YOLO format.

Model: YOLOv8, a deep learning-based object detection model.

Output: A system capable of detecting and classifying vehicles in real-time.

1. Dataset
The dataset used for this project is a custom vehicle detection dataset sourced from Roboflow. It contains images of different vehicles and their respective annotations.

Dataset Breakdown:
Number of Classes: 5 (Bus, Car, Motorcycle, Pickup, Truck)

Format: YOLO format (images and text files containing the bounding box annotations)

Image Size: Various resolutions, preprocessed to 640x640 for training.

Annotations: Bounding boxes with labels for each object (vehicle).

Dataset Structure:
bash
Copy
Edit
/train/images
/train/labels
/valid/images
/valid/labels
/test/images
/test/labels

2. Methodology
2.1. Preprocessing
Before training the model, the images were preprocessed:

Resizing images to a standard size (640x640 pixels).

Data augmentation (if applied) to improve model generalization.

2.2. Model Architecture
The model used is YOLOv8, an advanced version of the YOLO (You Only Look Once) architecture, which is optimized for real-time object detection tasks. It is particularly effective for detecting multiple objects in images.

Key Layers:

Convolutional Layers: Extract features from images.

Fully Connected Layers: Make predictions based on the extracted features.

Bounding Box Predictions: Determine object locations and class labels.

2.3. Training the Model
Pretrained Model: Used YOLOv8n.pt as a starting point.

Hyperparameters:

Epochs: 50

Batch Size: 16

Image Size: 640x640 pixels

Loss Functions: Combined box loss, classification loss, and distance loss for more accurate detection.

Training command:

python
Copy
Edit
from ultralytics import YOLO

# Initialize YOLOv8 model
model = YOLO('yolov8n.pt')  # Pretrained model

# Train the model
model.train(data='path/to/data.yaml', epochs=50, batch=16, imgsz=640)
2.4. Evaluation
The model was evaluated on:

Precision: How many of the predicted vehicles were correct.

Recall: How many of the actual vehicles were detected.

mAP50 (Mean Average Precision at IoU threshold of 50%): Indicates the overall accuracy of the model.

mAP50-95: A more stringent evaluation metric for object detection tasks.

3. Results
After training for 50 epochs, the YOLOv8 model was evaluated on the test set. The results were as follows:

Model Performance Metrics:
mAP50: 31% (Mean Average Precision at IoU=50%)

mAP50-95: 20% (Mean Average Precision across different IoU thresholds)

Example Detection:
The model was tested on images containing multiple vehicles, and it successfully detected and classified them.

Bounding boxes were drawn around detected vehicles, with corresponding class labels.

Challenges:
The dataset had some noisy labels, leading to a lower mAP.

Certain vehicle types, especially smaller vehicles like motorcycles, had lower detection accuracy.

4. Improvements
While the model is capable of detecting vehicles, there are still some areas that need improvement:

Data Augmentation: More aggressive data augmentation techniques (like rotation, scaling, and flipping) could improve model robustness.

Hyperparameter Tuning: Experimenting with other YOLO models (such as yolov8m or yolov8l) could yield better results.

Larger Dataset: The dataset could be expanded with more vehicle images and diverse scenarios.

Advanced Evaluation: Using more detailed metrics such as precision-recall curves could provide more insights into performance.

5. Conclusion
This project successfully demonstrated the use of YOLOv8 for vehicle detection. Despite some challenges in terms of model accuracy, it provides a good starting point for real-time object detection tasks. The project can be expanded with more data, fine-tuned hyperparameters, and enhanced preprocessing steps to improve performance further.

6. Future Work
Deploying the Model: Implement the model in a real-time vehicle detection application (e.g., traffic monitoring or autonomous driving).

Real-time Inference: Integrating the model with video streams for real-time detection.

Model Optimization: Optimizing the model for faster inference, such as converting it to TensorFlow Lite for mobile deployment.

7. Acknowledgments
YOLOv8 by Ultralytics: A fast and efficient object detection framework.

Roboflow: For providing the dataset for vehicle detection.

8. References
Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. CVPR.

Ultralytics. (2023). YOLOv8 Documentation. Retrieved from https://docs.ultralytics.com/

