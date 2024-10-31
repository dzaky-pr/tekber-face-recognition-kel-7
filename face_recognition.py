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
