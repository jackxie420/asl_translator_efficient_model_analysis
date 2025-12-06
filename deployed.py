import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

MODEL_PATH = "deploy_params.h5"
asl_model = tf.keras.models.load_model(MODEL_PATH)

alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", 
            "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

camera = cv2.VideoCapture(0)

hand_solution = mp.solutions.hands
hand_detector = hand_solution.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
drawing_utils = mp.solutions.drawing_utils

while True:
    ret, video_frame = camera.read()
    
    if not ret:
        continue
    
    rgb_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
    detection_results = hand_detector.process(rgb_frame)
    
    height, width, channels = video_frame.shape
    
    if detection_results.multi_hand_landmarks:
        for hand_landmarks in detection_results.multi_hand_landmarks:
            landmark_features = []
            for landmark in hand_landmarks.landmark:
                landmark_features.extend([landmark.x, landmark.y, landmark.z])
            
            model_input = np.array(landmark_features).reshape(-1, 63, 1)
            
            prediction = asl_model.predict(model_input, verbose=0)
            predicted_index = np.argmax(prediction)
            
            predicted_letter = alphabet[predicted_index]
            cv2.putText(
                video_frame, 
                predicted_letter, 
                (120, 170), 
                cv2.FONT_HERSHEY_PLAIN, 
                3, 
                (255, 0, 0), 
                3
            )
            
            drawing_utils.draw_landmarks(
                video_frame, 
                hand_landmarks, 
                hand_solution.HAND_CONNECTIONS
            )
    
    cv2.imshow("Image", video_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
