# Sign Language Detection

This project implements a deep learning model to detect and classify sign language gestures from various input sources, including images, videos, and live webcam streams. The model is deployed in an interactive Streamlit web application, providing users with an intuitive interface for real-time gesture recognition.

## Project Description

The application utilizes a pre-trained YOLO-NAS model tailored for sign language detection. The model was trained using a custom dataset following the approach outlined in [this project](https://github.com/MuhammadMoinFaisal/Computervisionprojects/blob/main/YOLO-NAS%20Sign%20Language%20Detection/Train_YOLONAS_Custom_Dataset_Sign_Language_Complete.ipynb) by [Muhammad Moin Faisal](https://github.com/MuhammadMoinFaisal). Building on top of this foundation, the model has been integrated into a comprehensive web platform that supports multiple input types for gesture detection.

### Key Features

- **Image Upload**: Users can upload an image, and the system will detect sign language gestures within the image.
- **Video Upload**: The model processes uploaded videos to identify gestures throughout the duration of the video.
- **Real-time Webcam Detection**: Users can activate their webcam to detect and classify gestures in real-time.
- **Homepage**: The app includes a user-friendly homepage to guide users through the available functionalities.

## Getting Started

### Prerequisites

- Python 3.7 - 3.9
- Streamlit
- SuperGradients
- OpenCV
- PyTorch
- Additional dependencies can be installed from the `requirements.txt` file.

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/darshneesunderraj/Sign-Language_Detection.git
    cd Sign-Language_Detection
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the trained YOLO-NAS model and place it in the `model_weights` folder:
    - Create a directory named `model_weights` in the project root if it doesn't already exist:
        ```bash
        mkdir model_weights
        ```
    - Download the pre-trained model weights file and save it in the `model_weights` directory as `ckpt_best.pth`:
        ```bash
        wget <url_to_your_trained_model> -O model_weights/ckpt_best.pth
        ```
    - Alternatively, manually download the file and move it into the `model_weights` folder.

4. Run the Streamlit application:
    ```bash
    streamlit run main.py
    ```

## Model Overview

The sign language detection model used in this project was originally trained using the approach outlined by [Muhammad Moin Faisal](https://github.com/MuhammadMoinFaisal). The model architecture is based on YOLO-NAS and has been fine-tuned to detect 26 unique sign language gestures. This project extends the model's functionality by deploying it in a web-based interface for broader accessibility and interaction.

## Credits

This project is inspired by the work on YOLO-NAS Sign Language Detection by [Muhammad Moin Faisal](https://github.com/MuhammadMoinFaisal), with significant adaptations to integrate the model into a fully functional Streamlit web application, allowing for image, video, and webcam-based gesture recognition.
