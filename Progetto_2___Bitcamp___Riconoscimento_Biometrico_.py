import cv2
import mediapipe as mp
import numpy as np
import os

# Disabilita i messaggi di log TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disabilita messaggi di livello INFO e DEBUG

# Percorsi delle immagini di riferimento
impronta_digitale_path = r"C:\Users\Gianni\Desktop\impronta digitale_reference.jpg"
retina_path = r"C:\Users\Gianni\Desktop\volto_reference.jpg"  # Questo ora non viene usato
voce_audio_path = r"C:\Users\Gianni\Desktop\verifica_voce_reference.wav"

# Configurazione di Mediapipe per il riconoscimento facciale
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Funzione per confrontare il volto acquisito con un volto di riferimento
def confronta_volto(volto_acquisito, volto_reference_path):
    # Carica l'immagine di riferimento del volto
    volto_reference = cv2.imread(volto_reference_path)
    volto_reference_rgb = cv2.cvtColor(volto_reference, cv2.COLOR_BGR2RGB)
    volto_acquisito_rgb = cv2.cvtColor(volto_acquisito, cv2.COLOR_BGR2RGB)

    # Usa il Face Mesh per ottenere i punti facciali
    volto_reference_landmarks = face_mesh.process(volto_reference_rgb).multi_face_landmarks
    volto_acquisito_landmarks = face_mesh.process(volto_acquisito_rgb).multi_face_landmarks

    if volto_reference_landmarks and volto_acquisito_landmarks:
        volto_reference_landmarks = volto_reference_landmarks[0].landmark
        volto_acquisito_landmarks = volto_acquisito_landmarks[0].landmark
        
        # Confronta la distanza tra i punti del volto
        differenza = 0
        for i in range(0, len(volto_reference_landmarks)):
            differenza += np.linalg.norm(
                np.array([volto_reference_landmarks[i].x, volto_reference_landmarks[i].y]) - 
                np.array([volto_acquisito_landmarks[i].x, volto_acquisito_landmarks[i].y])
            )
        
        # Se la differenza totale è inferiore a una soglia, considera il volto riconosciuto
        if differenza < 0.5:  # Soglia da adattare
            return True
    return False

# Cattura della videocamera con DirectShow come backend (per risolvere i problemi con MSMF)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Usa DirectShow come backend video

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Converti l'immagine in RGB per Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Verifica se sono stati trovati volti
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                # Disegna i punti sulla faccia
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Mostra la finestra di scansione del volto
        cv2.putText(frame, 'Verifica volto in corso...', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Estrai la zona del volto (puoi adattare questa parte in base alla tua applicazione)
        h, w, _ = frame.shape
        volto_acquisito = frame[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]  # Estrarre una porzione che contiene il volto

        # Confronta il volto acquisito con l'immagine di riferimento
        if confronta_volto(volto_acquisito, retina_path):  # Usa il volto_reference_path invece del retina_path
            print("Riconoscimento volto riuscito!")
            cv2.putText(frame, "Volto riconosciuto!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Scansione volto', frame)
            break  # Uscita dal ciclo dopo successo del riconoscimento
        else:
            cv2.putText(frame, "Volto non riconosciuto.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Visualizza l'immagine
    cv2.imshow('Scansione volto', frame)

    # Interrompi con 'q' o quando il riconoscimento è riuscito
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
