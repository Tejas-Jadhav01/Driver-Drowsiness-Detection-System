import numpy as np
import cv2
import cv2

def calculate_ear(eye_landmarks):
    """
    Calculates the Eye Aspect Ratio (EAR) for a single eye.
    The EAR is the ratio of the distance between the vertical eye landmarks 
    and the distance between the horizontal eye landmarks.

    Args:
        eye_landmarks (list): A list of 6 facial landmark coordinates for one eye.

    Returns:
        float: The calculated Eye Aspect Ratio.
    """
    # Vertical eye landmarks
    p2_p6 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    p3_p5 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])

    # Horizontal eye landmark
    p1_p4 = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

    # Compute the eye aspect ratio
    ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)

    return ear

def calculate_mar(mouth_landmarks):
    """
    Calculates the Mouth Aspect Ratio (MAR) to detect yawning.
    The MAR is the ratio of the distance between the vertical mouth landmarks (lips)
    and the distance between the horizontal mouth landmarks (corners).

    Args:
        mouth_landmarks (list): A list of facial landmark coordinates for the mouth.

    Returns:
        float: The calculated Mouth Aspect Ratio.
    """
    # Vertical mouth landmarks (upper and lower lip)
    p1 = mouth_landmarks[0]
    p2 = mouth_landmarks[1]
    
    # Horizontal mouth landmarks (corners)
    p3 = mouth_landmarks[2]
    p4 = mouth_landmarks[3]

    # Calculate the distances
    vertical_dist = np.linalg.norm(p1 - p2)
    horizontal_dist = np.linalg.norm(p3 - p4)

    # Compute the mouth aspect ratio
    mar = vertical_dist / horizontal_dist

    return mar

def estimate_head_pose(landmarks_2d, image_shape):
    """
    Estimates the head pose (pitch, yaw, roll) from 2D facial landmarks.

    Args:
        landmarks_2d (np.array): Array of 2D coordinates for specific facial landmarks.
        image_shape (tuple): The shape of the input image (height, width).

    Returns:
        tuple: A tuple containing the pitch, yaw, and roll angles in degrees.
    """
    # A generic 3D model of a face, assuming the center of the head is at the origin.
    # These points correspond to: Nose tip, Chin, Left eye left corner, Right eye right corner,
    # Left mouth corner, Right mouth corner.
    model_points_3d = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    focal_length = image_shape[1]
    center = (image_shape[1] / 2, image_shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    # Assuming no lens distortion
    dist_coeffs = np.zeros((4, 1))

    # Solve the PnP problem
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points_3d, landmarks_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    # Project a 3D point (e.g., 1000, 0, 0) onto the image plane.
    # We use this to draw a line sticking out of the nose.
    (nose_end_point2D, jacobian) = cv2.projectPoints(
        np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs
    )

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Decompose rotation matrix to get Euler angles
    sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0
        
    # Convert to degrees
    pitch = np.rad2deg(x)
    yaw = np.rad2deg(y)
    roll = np.rad2deg(z)

    return pitch, yaw, roll, nose_end_point2D
