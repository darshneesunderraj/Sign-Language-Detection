import streamlit as st
import imageio
import numpy as np
from super_gradients.training import models
from PIL import Image
import tempfile
import os

best_model = models.get('yolo_nas_s', num_classes=26, checkpoint_path='model_weights/ckpt_best.pth')
st.title("Sign Language Detection")

st.sidebar.image("images/logo.png", use_column_width=True)
st.sidebar.markdown(
    """
    <div style='text-align: center; font-size: 20px; font-weight: bold;'>
        Catch every gesture,<br>Translate with ease!
    </div>
    """, unsafe_allow_html=True
)

st.subheader("Get started by exploring the app ")

col1, col2, col3 = st.columns(3)

with col1:
    st.image("images/sample1.jpg", caption="Sample 1")
with col2:
    st.image("images/sample2.jpg", caption="Sample 2")
with col3:
    st.image("images/sample3.jpg", caption="Sample 3")

st.markdown("""
*What can you do?*
- Detect sign language gestures from images, videos, or live webcam.
- Real-time gesture detection with YOLO-NAS technology.
- Upload and analyze sign language videos with automatic gesture recognition.
""")

st.write("Ready to try? Click below to start detecting!")
if st.button("Start Detection"):
    st.session_state.detection_started = True

if st.session_state.get('detection_started', False):
    st.header("Let's do some detection!")
    detection_type = st.selectbox("Choose how you want to detect:", ("Choose an Option", "Image Upload", "Video Upload", "Webcam"))

    if detection_type == "Webcam":
        run_webcam = st.checkbox("Activate Webcam")

        if run_webcam:
            st.write("Webcam is active.")
            best_model.predict_webcam()

    elif detection_type == "Image Upload":
        uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width=True)

            with col2:
                results = best_model.predict(image)
                detected_image = results.draw()
                st.image(detected_image, caption="Detected Image", use_column_width=True)

    elif detection_type == "Video Upload":
        uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            temp_input_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_input_file.write(uploaded_video.read())
            temp_input_file.close()

            output_video_path = 'Video/output_video.mp4'

            # Ensure the output directory exists
            os.makedirs('Video', exist_ok=True)

            reader = imageio.get_reader(temp_input_file.name, 'ffmpeg')
            fps = reader.get_meta_data()['fps']
            writer = imageio.get_writer(output_video_path, fps=fps)

            st.write("Processing the video, please wait for a few minutes...")

            for frame in reader:
                frame_batch = np.expand_dims(frame, axis=0)
                results = best_model.predict(frame_batch)
                detected_image = results.draw()  # Ensure the closing parenthesis is added

                # Append the detected frame to the output video
                writer.append_data(np.array(detected_image))

            reader.close()
            writer.close()

            col1, col2 = st.columns(2)
            with col1:
                st.video(temp_input_file.name)
            with col2:
                st.video(output_video_path)

            try:
                os.remove(temp_input_file.name)
                os.remove(output_video_path)
            except PermissionError:
                st.write("Temporary files are still in use. Please restart the app to clear them.")

            st.write("Video processing complete!")

st.markdown("""
### Key Features:
- 🖼 *Image Upload:* Upload an image showing a sign language gesture.
- 📹 *Video Upload:* Analyze and translate sign language from video.
- 🎥 *Live Webcam Detection:* Detect gestures in real-time using your webcam.
""")
st.image("images/signs.jpg", caption="Sign Language Signs", use_column_width=True)

if st.button("Learn More"):
    st.header("About This App")

    col1, col2 = st.columns([1, 1])
    col3, col4 = st.columns([1, 1])

    with col1:
        st.write(""" 
            Sign Language is a powerful form of communication used by millions of people worldwide. 
            This app is designed to bridge the gap between spoken and sign languages by using advanced 
            AI models to detect and translate gestures in real-time.
        """)

    with col2:
        st.write(""" 
            ### How It Works
            The core of this app uses a deep learning model called YOLO-NAS, which is trained to detect 
            various hand gestures representing letters and numbers in sign language. You can upload an 
            image or video, or use your webcam, and the app will instantly recognize and translate the 
            gestures it detects.
        """)

    with col3:
        st.write(""" 
            ### Why It's Important
            For those who rely on sign language as their primary mode of communication, tools like this 
            are crucial. It can break down communication barriers and make interactions smoother for 
            both sign language users and non-users. 
        """)

    with col4:
        st.write(""" 
            ### How to Get Started
            You can start using the app by choosing one of the detection modes from the homepage. Whether 
            you upload an image, video, or use your webcam, our app will guide you through the process of 
            translating sign language gestures with ease. Ready to start detecting? Head back to the main 
            page and choose your detection mode.
        """)

st.markdown("<br>", unsafe_allow_html=True)