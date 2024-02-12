import cv2
import dlib
import numpy as np

# Load the pre-trained face detector and landmark predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Load the glasses image with an alpha channel (transparency)
glasses_img = cv2.imread("images/glasses.png", -1)

# Function to overlay glasses on the face
def overlay_glasses(frame, landmarks):
	# Calculate the position of the glasses
	glasses_width = int(np.linalg.norm(landmarks[36] - landmarks[45]) * 1.5)
	glasses_height = int(glasses_width * (glasses_img.shape[0] / glasses_img.shape[1]))

	# Resize the glasses image to match the calculated size
	glasses_resized = cv2.resize(glasses_img, (glasses_width, glasses_height))

	# Calculate the position to center the glasses
	x_offset = int(landmarks[27, 0] - glasses_width / 2)
	y_offset = int(landmarks[27, 1] - glasses_height / 2) + 10

	# Clip the overlay region to stay within the frame boundaries
	x1, x2 = max(x_offset, 0), min(x_offset + glasses_width, frame.shape[1])
	y1, y2 = max(y_offset, 0), min(y_offset + glasses_height, frame.shape[0])

	# Calculate the alpha values for blending
	alpha_glasses = glasses_resized[:, :, 3] / 255.0
	alpha_frame = 1.0 - alpha_glasses

	# Overlay the glasses on the frame
	try:
		frame[y1:y2, x1:x2, :3] = (alpha_glasses[:, :, np.newaxis] * glasses_resized[:, :, :3] + alpha_frame[:, :, np.newaxis] * frame[y1:y2, x1:x2, :3])
	except:
		pass

def face_filter_cam():
	# Open the webcam
	cap = cv2.VideoCapture(0)

	while True:
		ret, frame = cap.read()

		# Convert the frame to grayscale for face detection
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Detect faces in the frame
		faces = face_detector(gray)

		for face in faces:
			# Detect facial landmarks
			landmarks = landmark_predictor(gray, face)

			# Convert landmarks to NumPy array for easier indexing
			landmarks_np = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

			# Overlay the glasses on the face
			overlay_glasses(frame, landmarks_np)

		# Display the frame with the overlay
		cv2.imshow("Glasses Filter", frame)

		# Break the loop if 'q' key is pressed
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# Release the webcam and close all windows
	cap.release()
	cv2.destroyAllWindows()
