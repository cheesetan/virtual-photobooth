import cv2
import time
from cv2 import dnn
import numpy as np
from constants import threshold, center_variance, size_variance, image_std
from emotion_detection import predict
from anchor_boxes import convert_locations_to_boxes, center_form_to_corner_form, define_img_size

# Function to overlay emoji on detected faces
# Function to overlay emoji on detected faces
def overlay_emoji(img_ori, x1, y1, x2, y2, pred):
    # Define a dictionary to map emotion indices to labels
    emotion_dict = {
        0: 'neutral', 
        1: 'happiness', 
        2: 'surprise', 
        3: 'sadness',
        4: 'anger', 
        5: 'disgust', 
        6: 'fear'
    }

    # Default emoji image for neutral emotion
    imageText = "others/neutral.png"

    # Update the emoji image based on the predicted emotion
    if pred == "happiness":
        imageText = "others/happy.png"
    elif pred == "surprise":
        imageText = "others/surprise.png"
    elif pred == "sadness":
        imageText = "others/sadness.png"
    elif pred == "anger":
        imageText = "others/anger.png"
    elif pred == "disgust":
        imageText = "others/disgust.png"
    elif pred == "fear":
        imageText = "others/fear.png"

    # Read the emoji image with alpha channel (transparency)
    emotion_image = cv2.imread(imageText, cv2.IMREAD_UNCHANGED)

    # Ensure that emotion image size matches the ROI size
    emotion_image = cv2.resize(emotion_image, (x2 - x1, y2 - y1))

    # Compute the region of interest (ROI) in the original frame
    roi = img_ori[y1:y2, x1:x2]

    # Resize the ROI to match the emotion image size
    roi = cv2.resize(roi, (x2 - x1, y2 - y1))

    # Extract the alpha channel from the emotion image
    alpha_channel = emotion_image[:, :, 3] / 255.0

    # Blend the emotion image and the ROI with transparency
    blended_img = (1.0 - alpha_channel[:, :, np.newaxis]) * roi + alpha_channel[:, :, np.newaxis] * emotion_image[:, :, :3]

    # Convert the blended image to unsigned 8-bit integers
    blended_img = blended_img.astype(np.uint8)

    try:
        # Overlay the blended image on the original frame within the specified ROI
        img_ori[y1:y2, x1:x2] = blended_img
    except:
        # Handle any potential errors during overlay
        print("error")

# Function to perform live emotion detection from webcam
def emoji_cam():
    # Define emotion labels
    emotion_dict = {
        0: 'neutral', 
        1: 'happiness', 
        2: 'surprise', 
        3: 'sadness',
        4: 'anger', 
        5: 'disgust', 
        6: 'fear'
    }

    # Initialise webcam capture
    cap = cv2.VideoCapture(0)

    # Set up video recording parameters
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter('others/infer2-test.avi', 
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10, size)

    # Read pre-trained emotion recognition model
    model = cv2.dnn.readNetFromONNX('others/emotion-ferplus-8.onnx')
    
    # Read the Caffe face detector model
    model_path = 'RFB-320/RFB-320.caffemodel'
    proto_path = 'RFB-320/RFB-320.prototxt'
    net = dnn.readNetFromCaffe(proto_path, model_path)
    input_size = [320, 240]
    width = input_size[0]
    height = input_size[1]
    priors = define_img_size(input_size)

    # Main loop for live webcam emotion detection
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img_ori = frame
            rect = cv2.resize(img_ori, (width, height))
            rect = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
            net.setInput(dnn.blobFromImage(
                rect, 1 / image_std, (width, height), 127)
            )
            start_time = time.time()
            boxes, scores = net.forward(["boxes", "scores"])
            boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
            scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
            boxes = convert_locations_to_boxes(
                boxes, priors, center_variance, size_variance
            )
            boxes = center_form_to_corner_form(boxes)
            boxes, labels, probs = predict(
                img_ori.shape[1], 
                img_ori.shape[0], 
                scores, 
                boxes, 
                threshold
            )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Loop through detected boxes and perform emotion prediction
            for (x1, y1, x2, y2) in boxes:
                w = x2 - x1
                h = y2 - y1

                if gray[y1:y1 + h, x1:x1 + w].size == 0:
                    continue

                # Resize and preprocess face image for emotion prediction
                resize_frame = cv2.resize(
                    gray[y1:y1 + h, x1:x1 + w], (64, 64)
                )
                resize_frame = resize_frame.reshape(1, 1, 64, 64)
                model.setInput(resize_frame)
                output = model.forward()
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                # print(f"FPS: {fps:.1f}")

                # Predict emotion label based on model output
                pred = emotion_dict[list(output[0]).index(max(output[0]))]

                # Overlay emoji on the original frame
                overlay_emoji(img_ori, x1, y1, x2, y2, pred)

                # Write the frame with overlay to the video output
                result.write(frame)
        
            # Display the frame with overlay
            cv2.imshow('frame', frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release video capture and writer resources
    cap.release()
    result.release()
    cv2.destroyAllWindows()