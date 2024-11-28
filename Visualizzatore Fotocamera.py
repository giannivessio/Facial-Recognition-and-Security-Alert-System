import os
import numpy as np
import cv2
from deepface import DeepFace

# Carica il database degli embedding salvato
if not os.path.exists("embedding_database.npy"):
    raise FileNotFoundError("Il file embedding_database.npy non esiste!")

embedding_database = np.load("embedding_database.npy", allow_pickle=True)

# Funzione per calcolare la distanza tra due embedding
def calcola_distanza(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

# Cattura della videocamera
cap = cv2.VideoCapture(0)
print("Premi 'q' per uscire.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    try:
        # Estrazione dei volti nel frame
        faces = DeepFace.extract_faces(frame, detector_backend='opencv')

        if len(faces) > 0:
            # Salva il frame come immagine temporanea per DeepFace
            temp_path = "temp_frame.jpg"
            cv2.imwrite(temp_path, frame)

            # Calcola l'embedding del volto rilevato
            embedding_live = DeepFace.represent(temp_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]

            # Trova il nome con la distanza più bassa
            nome_rilevato = "Sconosciuto"
            distanza_minima = float("inf")

            for user in embedding_database:
                distanza = calcola_distanza(embedding_live, user["embedding"])
                if distanza < 0.6 and distanza < distanza_minima:  # Soglia di somiglianza
                    nome_rilevato = user["name"]
                    distanza_minima = distanza

            # Trova il volto nella scena e disegna un rettangolo
            face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray_frame, 1.1, 4)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Mostra il nome del riconoscimento sul frame
            cv2.putText(frame, f"Identificato: {nome_rilevato}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Volto non rilevato", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    except Exception as e:
        cv2.putText(frame, "Errore nel riconoscimento", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print(f"Errore: {e}")

    # Mostra il frame
    cv2.imshow("Riconoscimento facciale", frame)

    # Interrompi con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
