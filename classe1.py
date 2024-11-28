import cv2
import face_recognition
import numpy as np

# Percorsi dei file di riferimento
volto_reference_path = r"C:\Users\Gianni\Desktop\volto_reference.jpg"

# Carica l'immagine di riferimento e calcola gli encoding
volto_reference = face_recognition.load_image_file(volto_reference_path)
volto_reference_encoding = face_recognition.face_encodings(volto_reference)[0]

# Inizializza la videocamera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Converti il frame in RGB (face_recognition richiede RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Rileva i volti nel frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Confronta il volto acquisito con quello di riferimento
        matches = face_recognition.compare_faces([volto_reference_encoding], face_encoding)
        distance = face_recognition.face_distance([volto_reference_encoding], face_encoding)

        # Disegna un rettangolo attorno al volto
        top, right, bottom, left = face_location
        color = (0, 255, 0) if matches[0] else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Mostra se il volto è riconosciuto o meno
        label = "Riconosciuto" if matches[0] else "Non riconosciuto"
        cv2.putText(frame, f"{label} ({distance[0]:.2f})", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Mostra il frame
    cv2.imshow('Riconoscimento Facciale', frame)

    # Premi 'q' per uscire
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
