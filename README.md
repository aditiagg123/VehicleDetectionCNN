 🚗 Vehicle Detection Using YOLOv8

 1. Introduction
In this project, I implemented a **vehicle detection system** using **YOLOv8** (You Only Look Once version 8), a modern object detection model.  
The goal is to **identify and classify** five types of vehicles:
- Bus
- Car
- Motorcycle
- Pickup
- Truck


 2. Dataset
- **Source**: Dataset collected from **Roboflow**.
- **Classes**: 5 vehicle types.
- **Format**: YOLO format (images and text files with bounding box annotations).
- **Dataset Structure**:
  ```
  /train/images
  /train/labels
  /valid/images
  /valid/labels
  /test/images
  /test/labels
  ```
- **Image Size**: Resized to **640x640 pixels** for training.


 3. Preprocessing
- **Resizing**: All images resized to 640x640.
- **Data Augmentation**: Minor transformations like flipping and scaling were used to make the model robust.


 4. Model Architecture
- **Model Used**: YOLOv8n (nano version for faster training).
- **Key Components**:
  - Convolutional Layers for feature extraction.
  - Bounding Box Head for object detection.
  - Classification Head for predicting the class of each detected object.



 5. Training Details
- **Base Model**: Pretrained weights from `yolov8n.pt`.
- **Training Hyperparameters**:
  - **Epochs**: 50
  - **Batch Size**: 16
  - **Image Size**: 640x640
- **Training Command**:
  ```python
  from ultralytics import YOLO
  
  model = YOLO('yolov8n.pt')
  model.train(data='path/to/data.yaml', epochs=50, batch=16, imgsz=640)
  ```

---

 6. Evaluation
    
- **Metrics Used**:
  - **mAP50**: 31% (Mean Average Precision at IoU = 50%)
  - **mAP50-95**: 20% (Across different IoU thresholds)
- **Performance Observations**:
  - Good detection of large vehicles like Bus and Truck.
  - Lower accuracy for smaller vehicles like Motorcycle.



 7. Testing
- **True Labels**: Extracted from test image folder structure.
- **Predictions**: Model predictions were compared with true labels.
- **Confusion Matrix**: Created to visualize the performance.
- **Accuracy Calculation**: Overall detection and classification accuracy was calculated.

---

 8. Challenges Faced
- Some images in the dataset had **noisy labels** or **wrong bounding boxes**.
- **Smaller objects** like motorcycles were harder to detect compared to bigger vehicles.
- **Model Generalization**: The model may not perform perfectly on completely new unseen environments.



 9. Improvements and Future Work
- **Data Augmentation**: Use more aggressive techniques like rotation, color jittering, etc.
- **Hyperparameter Tuning**: Try different batch sizes, learning rates, and other YOLOv8 variants like YOLOv8m or YOLOv8l.
- **Larger Dataset**: Collect more data with more diverse environments.
- **Real-time Deployment**: Implement real-time detection on live video streams.
- **Model Optimization**: Export the model to a lightweight format like TensorFlow Lite for mobile deployment.



 10. Conclusion
This project successfully demonstrates how YOLOv8 can be applied to **real-time vehicle detection and classification**.  
Even though there are areas for improvement, the model shows **good initial performance** and can be **further enhanced** by better data and tuning.



 11. Acknowledgments
- **Ultralytics YOLOv8**: For the powerful object detection framework.
- **Roboflow**: For providing the labeled dataset.

---

 12. References
- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection.* CVPR.
- Ultralytics. (2023). *YOLOv8 Documentation.* Retrieved from [https://docs.ultralytics.com/](https://docs.ultralytics.com/)

