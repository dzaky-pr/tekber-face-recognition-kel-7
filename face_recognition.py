# import cv2

# recognizer = cv2.face.LBPHFaceRecognizer.create()
# recognizer.read("face-model.yml") # face model from face_training-py
# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# font = cv2.FONT_HERSHEY_COMPLEX

# id = 0
# names = ['None', 'Dzaky']
# cap = cv2.VideoCapture(0)

# while True:
#     _, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

#     for (x,y,w,h) in faces:
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
#         id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
#         if confidence < 100:
#             id = names[id]
#         else:
#             id = "unknown"
#         confidence = "{}%".format(round(100-confidence))
        
#         cv2.putText(frame, str(id), (x+5, y-5), font, 1, (255,0,0), 1)
#         cv2.putText(frame, str(confidence), (x+5, y+h-5), font, 1, (255,255,0), 1)

#     cv2.imshow( "Camera", frame) 
#     if cv2.waitKey(1) == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

# ! BATAS BATAS BATAS

# import cv2
# from keras.models import load_model
# import numpy as np
# from keras.models import load_model

# # Load gender classification model
# try:
#     gender_model = load_model('./gender_model.h5')  # Ensure you have a trained gender model
# except Exception as e:
#     print(f"Error loading model: {e}")

# # Load gender classification model
# # gender_model = load_model('gender_model.h5')  # Ensure you have a trained gender model

# recognizer = cv2.face.LBPHFaceRecognizer.create()
# recognizer.read("face-model.yml") 
# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# font = cv2.FONT_HERSHEY_COMPLEX

# names = ['None', 'Dzaky']
# gender_labels = ["Male", "Female"]

# cap = cv2.VideoCapture(0)

# while True:
#     _, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

#     for (x,y,w,h) in faces:
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = frame[y:y+h, x:x+w]
#         id, confidence = recognizer.predict(roi_gray)
        
#         if confidence < 100:
#             id = names[id]
#         else:
#             id = "unknown"
        
#         # Gender prediction
#         face_img = cv2.resize(roi_color, (96, 96))
#         face_img = face_img / 255.0
#         face_img = np.reshape(face_img, (1, 96, 96, 3))
#         gender = gender_labels[np.argmax(gender_model.predict(face_img))]

#         confidence_text = "{}%".format(round(100-confidence))
#         cv2.putText(frame, f"{id}, {gender}", (x+5, y-5), font, 1, (255,0,0), 1)
#         cv2.putText(frame, confidence_text, (x+5, y+h-5), font, 1, (255,255,0), 1)

#     cv2.imshow("Camera", frame) 
#     if cv2.waitKey(1) == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

# ! BATAS BATAS BATAS

# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model

# # Muat model gender dan emosi
# gender_model = load_model('model/Gender_model.h5')
# emotion_model = load_model('model/emotion_model.h5')

# # Inisialisasi recognizer untuk face recognition
# recognizer = cv2.face.LBPHFaceRecognizer.create()
# recognizer.read("face-model.yml")
# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# font = cv2.FONT_HERSHEY_COMPLEX

# id = 0
# names = ['None', 'Dzaky']
# gender_labels = ['Laki-laki', 'Perempuan']
# emotion_labels = ['Happy', 'Sad', 'Neutral']
# cap = cv2.VideoCapture(0)

# while True:
#     _, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         # Prediksi ID wajah dengan recognizer
#         id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
#         id = names[id] if confidence < 100 else "tidak dikenal"
#         confidence_text = "{}%".format(round(100 - confidence))

#         # Ekstraksi ROI (Region of Interest) untuk prediksi gender dan emosi
#         face_roi_gray = gray[y:y+h, x:x+w]

#         # Preprocessing untuk model gender
#         face_roi_color = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)  # Konversi ke RGB
#         face_roi_color = cv2.resize(face_roi_color, (256, 256))  # Resize ke (256, 256)
#         face_roi_color = face_roi_color.reshape(1, 256, 256, 3) / 255.0  # Normalisasi

#         # Prediksi gender
#         gender_pred = gender_model.predict(face_roi_color)
#         gender = gender_labels[np.argmax(gender_pred)]

#         # Preprocessing untuk model emosi
#         emotion_roi = cv2.resize(face_roi_gray, (48, 48))
#         emotion_roi = emotion_roi.reshape(1, 48, 48, 1) / 255.0
#         emotion_pred = emotion_model.predict(emotion_roi)
#         emotion = emotion_labels[np.argmax(emotion_pred)]

#         # Tampilkan hasil pada frame dengan outline putih dan teks hitam
#         text = f"{id}, {gender}, {emotion}"
        
#         # Outline putih
#         cv2.putText(frame, text, (x + 5, y - 25), font, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
#         cv2.putText(frame, confidence_text, (x + 5, y + h - 5), font, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
        
#         # Teks hitam di atas outline
#         cv2.putText(frame, text, (x + 5, y - 25), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#         cv2.putText(frame, confidence_text, (x + 5, y + h - 5), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

#     cv2.imshow("Camera", frame)
#     if cv2.waitKey(1) == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

# ! BATAS BATAS BAST
import cv2
from deepface import DeepFace

# Inisialisasi recognizer untuk face recognition
# recognizer = cv2.face.LBPHFaceRecognizer.create()
# recognizer.read("face-model.yml")
# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# font = cv2.FONT_HERSHEY_COMPLEX

# id = 0
# names = ['None', 'Dzaky']
# cap = cv2.VideoCapture(0)

# while True:
#     _, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         # Prediksi ID wajah dengan recognizer
#         id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
#         id = names[id] if confidence < 100 else "tidak dikenal"
#         confidence_text = "{}%".format(round(100 - confidence))

#         # Ekstraksi wajah untuk analisis
#         face_roi = frame[y:y+h, x:x+w]

#         # Analisis menggunakan DeepFace
#         analysis = DeepFace.analyze(face_roi, actions=['emotion', 'gender', 'age'], enforce_detection=False)

#         # Ambil hasil
#         gender = analysis[0]['gender']
#         emotion = analysis[0]['dominant_emotion']
#         age = analysis[0]['age']

#         # Tampilkan hasil pada frame dengan outline putih dan teks hitam
#         text = f"{id}, {gender}, {emotion}, {age} years"

#         # Outline putih
#         cv2.putText(frame, text, (x + 5, y - 25), font, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
#         cv2.putText(frame, confidence_text, (x + 5, y + h - 5), font, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
        
#         # Teks hitam di atas outline
#         cv2.putText(frame, text, (x + 5, y - 25), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
#         cv2.putText(frame, confidence_text, (x + 5, y + h - 5), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

#     cv2.imshow("Camera", frame)
#     if cv2.waitKey(1) == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
from deepface import DeepFace

# Inisialisasi recognizer untuk face recognition
recognizer = cv2.face.LBPHFaceRecognizer.create()
recognizer.read("face-model.yml")
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_COMPLEX

id = 0
names = ['None', 'Dzaky']
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Prediksi ID wajah dengan recognizer
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        id = names[id] if confidence < 100 else "tidak dikenal"
        confidence_text = "{}%".format(round(100 - confidence))

        # Ekstraksi wajah untuk analisis
        face_roi = frame[y:y+h, x:x+w]

        # Analisis menggunakan DeepFace
        analysis = DeepFace.analyze(face_roi, actions=['emotion', 'gender', 'age'], enforce_detection=False)

        # Ambil hasil
        dominant_gender = analysis[0]['gender']
        emotion = analysis[0]['dominant_emotion']
        age = analysis[0]['age']

        # Ambil akurasi gender
        gender_probs = analysis[0]['gender']
        # Ambil gender dengan akurasi tertinggi
        gender, probability = max(gender_probs.items(), key=lambda item: item[1])

        # Tampilkan hasil pada frame dengan outline putih dan teks hitam
        text = f"{id}, {gender}, {emotion}, {age} years"

        # Outline putih
        cv2.putText(frame, text, (x + 5, y - 25), font, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, confidence_text, (x + 5, y + h - 5), font, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
        
        # Teks hitam di atas outline
        cv2.putText(frame, text, (x + 5, y - 25), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, confidence_text, (x + 5, y + h - 5), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
