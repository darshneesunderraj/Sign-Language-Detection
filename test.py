import imageio
import numpy as np
from super_gradients.training import models

# Load the best YOLO-NAS model
best_model = models.get('yolo_nas_s',
                         num_classes=26,  # Adjust to your number of classes
                         checkpoint_path='model_weights/ckpt_best.pth')

# Define input and output video paths
input_video_path = 'Video/video-1.mp4'
output_video_path = 'Video/video-1-output.mp4'

# Open the input video with imageio
reader = imageio.get_reader(input_video_path, 'ffmpeg')
fps = reader.get_meta_data()['fps']

# Create a video writer to save the processed frames
writer = imageio.get_writer(output_video_path, fps=fps)

# Process each frame from the video
for frame in reader:
    # Convert the frame to the format needed (add batch dimension)
    frame_batch = np.expand_dims(frame, axis=0)  # Add batch dimension (1, H, W, C)

    # Use the model to predict on the frame
    results = best_model.predict(frame_batch)

    # Draw the predictions on the frame
    detected_image = results.draw()  # This draws the boxes and labels on the image

    # Write the detected frame back to the output video
    writer.append_data(detected_image)

# Close the writer after processing all frames
writer.close()

print(f"Processed video saved as {output_video_path}")
