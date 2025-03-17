import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from playsound import playsound
import threading

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

sound_on = False

def play_sound(sound_file):
   playsound(sound_file)

def get_hand_gesture(hand_landmarks):
    if hand_landmarks is None:
        return "No Hand"
    
    # Get y coordinates of finger tips and pips
    thumb_tip = hand_landmarks.landmark[4].y
    thumb_mcp = hand_landmarks.landmark[2].y
    index_tip = hand_landmarks.landmark[8].y
    index_pip = hand_landmarks.landmark[6].y
    middle_tip = hand_landmarks.landmark[12].y
    middle_pip = hand_landmarks.landmark[10].y
    ring_tip = hand_landmarks.landmark[16].y
    ring_pip = hand_landmarks.landmark[14].y
    pinky_tip = hand_landmarks.landmark[20].y
    pinky_pip = hand_landmarks.landmark[18].y

    # Check if fingers are raised
    fingers = []
    # Thumb
    fingers.append(thumb_tip < thumb_mcp)
    # Other fingers
    fingers.append(index_tip < index_pip)
    fingers.append(middle_tip < middle_pip)
    fingers.append(ring_tip < ring_pip)
    fingers.append(pinky_tip < pinky_pip)

    # Interpret gesture
    if all(fingers):
        return "OPEN HAND"
    elif not any(fingers):
        return "FIST"
    elif fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
        return "PEACE"
    elif fingers[1] and not any(fingers[2:]):
        return "POINTING"
    return "OTHER"

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cam.isOpened():
        ret, frame = cam.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))

        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Detect and display gestures
        right_hand_gesture = get_hand_gesture(results.right_hand_landmarks)
        left_hand_gesture = get_hand_gesture(results.left_hand_landmarks)
        
        # Display gestures on image
        cv2.putText(image, f"Right: {right_hand_gesture}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Left: {left_hand_gesture}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Full Body Detection', image)

        if right_hand_gesture == "PEACE" and sound_on == False:
            sound_on = True
            thread = threading.Thread(
                target=play_sound,
                args=('./sounds/cat.mp3',),
            )
            thread.start()
        if right_hand_gesture == "OPEN_HAND" and sound_on == False:
            sound_on = True
            thread = threading.Thread(
                target=play_sound,
                args=('./sounds/bell.mp3',),
            )
            thread.start()
        if right_hand_gesture == "POINTING" and sound_on == False:
            sound_on = True
            thread = threading.Thread(
                target=play_sound,
                args=('./sounds/wind.mp3',),
            )
            thread.start()

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()
