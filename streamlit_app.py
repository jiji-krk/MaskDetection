import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import torch
import numpy as np

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select Page", ["Exploratory Data Analysis", "YOLOv5 Results", "Real-Time Detection"])

if page == "Exploratory Data Analysis":

    st.title("Mask Detection Dashboard - EDA")
    st.write("""
    This page provides insights into the dataset, including class distributions and annotated images.
    """)

    # Load dataset
    annotations_file = r"mask_dataset/train/_annotations.csv"
    data = pd.read_csv(annotations_file)

    # Class distribution
    st.header("Class Distribution")
    class_counts = data["class"].value_counts()
    fig, ax = plt.subplots()
    class_counts.plot(kind="pie", ax=ax, title="Class Distribution", autopct='%1.1f%%')
    st.pyplot(fig)

    # Counts per file
    st.header("Annotations per File")
    file_counts = data.groupby(['filename', 'class']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    file_counts.plot(kind="bar", ax=ax, stacked=False, title="Annotations per Filename")
    plt.xticks(ticks=[], labels=[], rotation=90)
    st.pyplot(fig)

    st.write("The bar chart shows the number of annotations per filename for two classes: mask (blue) and no-mask (orange). There is a significant imbalance, with some files containing far more annotations (notably one with around 100 for mask), suggesting uneven data distribution across the dataset.")
    st.divider()

    # Identify filenames with only "mask" or "no-mask"
    st.header("Files with Only Mask or No-Mask Annotations")
    unique_class_per_file = data.groupby('filename')['class'].unique()
    only_mask = unique_class_per_file.apply(lambda x: len(x) == 1 and 'mask' in x).sum()
    only_no_mask = unique_class_per_file.apply(lambda x: len(x) == 1 and 'no-mask' in x).sum()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(['Only Mask', 'Only No-Mask'], [only_mask, only_no_mask], color=['green', 'red'])
    ax.set_title('Filenames with Only Mask or Only No-Mask Annotations')
    ax.set_ylabel('Count of Filenames')
    plt.tight_layout()
    st.pyplot(fig)

    st.write("The chart shows that there is no files having exclusively no-mask annotations.")
    st.divider()

    # Annotated images
    st.header("Annotated Images")
    col1, col2 = st.columns(2)
    with col1:
        st.image("bounding_box_image1.png", caption="Bounding Box Example 1", use_container_width=True)
    with col2:
        st.image("bounding_box_image2.png", caption="Bounding Box Example 2", use_container_width=True)

    st.write("We drew the bounding boxes by reading the image and its corresponding label file. Each line in the label file specifies the class (mask or no-mask) and the normalized coordinates of the bounding box (center_x, center_y, width, height). These coordinates were converted to pixel values and used to draw rectangles with OpenCV, where green boxes represent 'mask' and red boxes represent no-mask")


elif page == "YOLOv5 Results":
    # Page 2: YOLOv5 Results
    st.title("Mask Detection Dashboard - YOLOv5 Results")
    st.write("""
    This page displays the performance of the YOLOv5 model, including metrics and annotated results.
    """)

    # Dropdown to select dataset
    dataset_choice = st.selectbox("Choose the dataset to view results:", ["Validation Set", "Test Set", "Training Set"])

    if dataset_choice == "Validation Set":
        st.subheader("Results on Validation Set")
        
        # Display Confusion Matrix
        validation_confusion_matrix_path = "C:/Users/jinan/OneDrive/Documents/A5_DIA2/ML_CV/Projet/yolov5/runs/val/exp/confusion_matrix.png"
        st.image(validation_confusion_matrix_path, caption="Confusion Matrix - Validation Set", use_container_width=True)
        st.write("""
        This confusion matrix shows the model's prediction performance for three classes: **mask**, **no-mask**, and **background**. 
        - The diagonal cells represent correct predictions, such as 85% accuracy for detecting "mask". 
        - The model struggles to distinguish "no-mask" (55% accuracy) and shows confusion between "mask" and "no-mask" as well as with the "background". 
        Improving the model's capacity to reduce these misclassifications can enhance overall performance.
        """)
        st.divider()

        
        # Display Metrics
        metrics = ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95']
        values = [0.844, 0.684, 0.825, 0.544]
        
        bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.3f}', ha='center', va='bottom')
        
        plt.title('Performance Metrics: Validation Set')
        plt.ylabel('Scores')
        plt.xlabel('Metrics')
        st.pyplot(plt)

        st.write("""
        **Analysis of Performance Metrics:**
        - **Precision (0.844):** The model makes mostly accurate positive predictions.
        - **Recall (0.684):** Some actual positives are missed, suggesting room for improvement in detecting all relevant instances.
        - **mAP@0.5 (0.825):** The model performs well at detecting objects with 50% overlap between predicted and ground truth boxes.
        - **mAP@0.5:0.95 (0.544):** Performance decreases with stricter IoU thresholds, indicating challenges with precise localization.
        """)



    elif dataset_choice == "Test Set":
        st.subheader("Results on Test Set")
        
        # Display Confusion Matrix
        test_confusion_matrix_path = r"C:/Users/jinan/OneDrive/Documents/A5_DIA2/ML_CV/Projet/yolov5/runs/val/exp3/confusion_matrix.png"
        st.image(test_confusion_matrix_path, caption="Confusion Matrix - Test Set", use_container_width=True)
        st.write("""
        This confusion matrix shows the model's prediction performance for three classes: **mask**, **no-mask**, and **background**.  
        - The diagonal cells represent correct predictions, such as **91% accuracy** for detecting "mask".  
        - The model has significant confusion between "mask" and "no-mask" (e.g., **60% of actual "no-mask" predictions are misclassified as "mask").  
        - The "background" class has fewer misclassifications, but some overlap exists with "mask" (e.g., **5% confusion**).  
        Improving the model's precision and recall for the "no-mask" class could enhance its overall detection accuracy.
        """)
        st.divider()
        
        # Display Metrics
        metrics = ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95']
        values = [0.618, 0.867, 0.723, 0.437]
        
        bars = plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{value:.3f}', ha='center', va='bottom')
        
        plt.title('Performance Metrics: Test Set')
        plt.ylabel('Scores')
        plt.xlabel('Metrics')
        st.pyplot(plt)

        st.write("""
        **Analysis of Performance Metrics (Test Set):**
        - **Precision (0.618):** The model correctly identifies a moderate proportion of positive predictions, but there is room to improve false positive reduction.
        - **Recall (0.867):** The model captures most of the actual positive cases, showing strong detection capability.
        - **mAP@0.5 (0.723):** The model performs well for objects with a 50% overlap threshold between predictions and ground truth boxes.
        - **mAP@0.5:0.95 (0.437):** Performance drops significantly for stricter IoU thresholds, indicating difficulty in precise object localization.
        """)



    elif dataset_choice == "Training Set":
        st.subheader("Results on Training Set")
        
        # Read the results file
        results_file = r"C:/Users/jinan/OneDrive/Documents/A5_DIA2/ML_CV/Projet/yolov5/runs/train/exp/results.csv"
        results = pd.read_csv(results_file)

        # Clean column names
        results.columns = results.columns.str.strip()

        # Plot metrics over epochs
        epochs = results.index + 1
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, results['metrics/precision'], label='Precision', marker='o')
        plt.plot(epochs, results['metrics/recall'], label='Recall', marker='o')
        plt.plot(epochs, results['metrics/mAP_0.5'], label='mAP@0.5', marker='o')
        plt.plot(epochs, results['metrics/mAP_0.5:0.95'], label='mAP@0.5:0.95', marker='o')

        plt.title('Performance Metrics Over Epochs (Training Set)')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid()
        st.pyplot(plt)

        st.write("The curves represent the improvement of the model as training progresses. Higher values for these metrics indicate better detection accuracy")
        st.divider()

        # Display the training results image
        results_image_path = r"C:/Users/jinan/OneDrive/Documents/A5_DIA2/ML_CV/Projet/yolov5/runs/train/exp/results.png"
        st.image(results_image_path, caption="Training Results Overview", use_container_width=True)

        st.write("""
        - **Box Loss**: Checks how accurately the model predicts the location of objects' bounding boxes.
        - **Objectness Loss**: Measures if the model detects the presence or absence of objects correctly.
        - **Classification Loss**: Evaluates how well the model classifies detected objects into the right categories (mask, no-mask, background).
        - **Metrics**: Include precision, recall, and mAP, which show the overall accuracy of the model's predictions.
        """)


elif page == "Real-Time Detection":
    # Page 3: Real-Time Detection
    st.title("Real-Time Mask Detection")
    st.write("""
    This page uses your webcam to perform real-time mask detection with YOLOv5.
    """)

    # Load YOLOv5 model
    MODEL_PATH = "C:/Users/jinan/OneDrive/Documents/A5_DIA2/ML_CV/Projet/yolov5/runs/train/exp/weights/best.pt" #r"C:\Users\jinan\OneDrive\Documents\A5_DIA2\ML_CV\Projet\model.pt"  
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)

    st.write("""
    <span style="color:red;">
    For accurate detection, it is recommended to position yourself slightly farther from the camera, as the training images were primarily of individuals captured from a distance.
    </span>
    """, unsafe_allow_html=True)
    st.divider()

    # Start Video Capture
    run = st.checkbox("Activate Camera")
    FRAME_WINDOW = st.image([])  # Placeholder for video frames

    cap = cv2.VideoCapture(0)  # Open the default camera (ID 0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video")
            break

        # Convert frame to RGB (required by Streamlit)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLOv5 detection
        results = model(rgb_frame)
        detection_frame = np.squeeze(results.render())  # Get annotated frame

        # Update the Streamlit frame
        FRAME_WINDOW.image(detection_frame)

    cap.release()  # Release the camera
    st.write("Camera stopped.")
