import cv2
import dlib
import numpy as np
from enum import Enum
import global_variables

class FacePosition(Enum):
	EYES = 27
	NOSE = 33
	HEAD = 23

# Load the pre-trained face detector and landmark predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Function to overlay glasses on the face
# Tristan did the resizing, Klfton did the rotation to match angle head it tilted at
def overlay_eyes(image, frame, landmarks, position):
	# Calculate the position of the glasses
	glasses_width = int(np.linalg.norm(landmarks[36] - landmarks[45]) * 1.5)
	glasses_height = int(glasses_width * (image.shape[0] / image.shape[1]))

	# Resize the glasses image to match the calculated size
	glasses_resized = cv2.resize(image, (glasses_width, glasses_height))

	# Calculate the angle between landmarks[36] and landmarks[45] in radians
	angle_radians = -1 * np.arctan2(landmarks[45][1] - landmarks[36][1], landmarks[45][0] - landmarks[36][0])

	# Convert the angle from radians to degrees
	angle_degrees = np.degrees(angle_radians)

	# Calculate the center of the glasses
	glasses_center = (glasses_width // 2, glasses_height // 2)

	# Calculate the position to center the glasses
	x_offset = int(landmarks[position.value, 0] - glasses_width / 2)
	y_offset = int(landmarks[position.value, 1] - glasses_height / 2) + 10

	# Clip the overlay region to stay within the frame boundaries
	x1, x2 = max(x_offset, 0) + 100, min(x_offset + glasses_width, frame.shape[1])
	y1, y2 = max(y_offset, 0) + 100, min(y_offset + glasses_height, frame.shape[0])

	# Rotate the glasses to match the angle between landmarks[36] and landmarks[45]
	# center = (int(landmarks[position.value, 0]), int(landmarks[position.value, 1]))
	rotation_matrix = cv2.getRotationMatrix2D(glasses_center, angle_degrees, 1.0)
	rotated_overlay = cv2.warpAffine(glasses_resized, rotation_matrix, (glasses_width, glasses_height))

	# Calculate the alpha values for blending
	alpha_channel = rotated_overlay[:, :, 3] / 255.0
	alpha_frame = 1.0 - alpha_channel

	# Overlay the glasses on the frame
	try:
		if x1 < x2 and y1 < y2:
			frame[y1:y2, x1:x2, :3] = (alpha_channel[:, :, np.newaxis] * rotated_overlay[:, :, :3] + alpha_frame[:, :, np.newaxis] * frame[y1:y2, x1:x2, :3])
		else: 
			print("There was an unexpected error")
	except IndexError as e:
		print("IndexError: {}".format(e))
	except Exception as ex:
		print("An unexpected error occurred: {}".format(ex))
		

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

			# Load the glasses image with an alpha channel (transparency)
			mustache_img = cv2.imread("images/moustache.png", -1)
			glasses_img = cv2.imread("images/glasses.png", -1)
	
			# Overlay the filters on the face
			for filter in global_variables.filters_chosen:
				if filter == global_variables.Filters.GLASSES:
					overlay_eyes(glasses_img, frame, landmarks_np, FacePosition.EYES)
				elif filter == global_variables.Filters.MOUSTACHE:
					overlay_eyes(mustache_img, frame, landmarks_np, FacePosition.NOSE)

		# Display the frame with the overlay
		cv2.imshow("Glasses Filter", frame)

		# Break the loop if 'q' key is pressed
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# Release the webcam and close all windows
	cap.release()
	cv2.destroyAllWindows()
