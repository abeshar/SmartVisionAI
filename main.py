import cv2
import mediapipe as mp
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

predictions = []

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            data = []
            for lm in hand_landmarks.landmark:
                data.append(lm.x)
                data.append(lm.y)

            prediction = model.predict([np.array(data)])
            predictions.append(prediction[0])

            if len(predictions) > 10:
                predictions.pop(0)

            final_pred = max(set(predictions), key=predictions.count)

            cv2.putText(frame, final_pred, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)

    else:
        cv2.putText(frame, "No hand detected", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Sign Language", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()