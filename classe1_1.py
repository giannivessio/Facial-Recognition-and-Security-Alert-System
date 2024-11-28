import cv2
import face_recognition
import os
from screeninfo import get_monitors

# Ottieni la risoluzione dello schermo
monitor = get_monitors()[0]  # Ottieni il primo monitor
screen_width = monitor.width
screen_height = monitor.height

# Percorso della cartella principale che contiene le sottocartelle con le immagini
main_directory = r"C:\Users\Gianni\Desktop\Foto_Riconoscimento"  # Cambia questo percorso con la tua cartella principale

# Liste per memorizzare gli encoding dei volti, i nomi associati e i dati
known_face_encodings = []
known_face_names = []
person_data = {}

# Esplora tutte le sottocartelle nella cartella principale
for person_name in os.listdir(main_directory):
    person_folder = os.path.join(main_directory, person_name)
    
    # Verifica che sia una cartella
    if os.path.isdir(person_folder):
        # Aggiungi i dati della persona leggendo il file .txt
        data_file = os.path.join(person_folder, f"Dati {person_name}.txt")
        if os.path.isfile(data_file):
            with open(data_file, 'r') as file:
                person_info = file.read().splitlines()  # Suddividi il testo in linee
            person_data[person_name] = person_info
        else:
            person_data[person_name] = ["Dati non disponibili"]

        # Esplora tutte le immagini nella cartella della persona
        for image_file in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_file)
            
            # Controlla se il file è un'immagine (estensione .jpg, .png, .jpeg, ecc.)
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Carica l'immagine e ottieni gli encoding del volto
                try:
                    volto_image = face_recognition.load_image_file(image_path)
                    volto_encoding = face_recognition.face_encodings(volto_image)
                    
                    # Se trovi un volto nell'immagine, aggiungi l'encoding e il nome
                    if volto_encoding:
                        known_face_encodings.append(volto_encoding[0])
                        known_face_names.append(person_name)
                except Exception as e:
                    print(f"Errore nel caricare l'immagine {image_path}: {e}")

# Impostazione iniziale della finestra della videocamera
window_width = 1280  # Imposta la larghezza iniziale
window_height = 720  # Imposta l'altezza iniziale

# Inizializza la videocamera
cap = cv2.VideoCapture(0)

# Soglia di precisione per il riconoscimento (distanza inferiore a 0.55 per maggiore precisione)
precision_threshold = 0.55  # Soglia di somiglianza, più bassa è, più preciso è il riconoscimento

# Imposta la risoluzione iniziale della videocamera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)

# Crea una finestra (non a schermo intero)
cv2.namedWindow("Riconoscimento Facciale", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Riconoscimento Facciale", window_width, window_height)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Ottieni la dimensione corrente della finestra (in base alla finestra ridimensionata)
    window_width, window_height = cv2.getWindowImageRect("Riconoscimento Facciale")[2:4]

    # Adatta la risoluzione della videocamera alla finestra
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)

    # Converti il frame in RGB (face_recognition richiede RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Rileva i volti nel frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Confronta il volto acquisito con i volti di riferimento
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        distance = face_recognition.face_distance(known_face_encodings, face_encoding)

        # Determina il nome della persona se il volto è riconosciuto
        name = "Sconosciuto"
        person_info = []
        if True in matches:
            first_match_index = matches.index(True)
            # Se la distanza è inferiore alla soglia, consideriamo la corrispondenza
            if distance[first_match_index] < precision_threshold:
                name = known_face_names[first_match_index]
                person_info = person_data.get(name, ["Dati non disponibili"])

        # Ottieni i valori della posizione del volto (top, right, bottom, left)
        top, right, bottom, left = face_location

        # Aggiungi margine per allargare il rettangolo attorno al volto
        margin = 0.2  # Aggiungi un margine del 20% al rettangolo

        # Calcola il margine per ciascun lato
        width = right - left
        height = bottom - top
        left = int(left - margin * width)
        right = int(right + margin * width)
        top = int(top - margin * height)
        bottom = int(bottom + margin * height)

        # Assicurati che il rettangolo rimanga nei limiti dell'immagine
        left = max(left, 0)
        right = min(right, frame.shape[1])
        top = max(top, 0)
        bottom = min(bottom, frame.shape[0])

        # Disegna un rettangolo attorno al volto
        color = (0, 255, 0) if name != "Sconosciuto" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Mostra il nome e la distanza se il volto è riconosciuto o meno
        label = f"{name} ({distance[first_match_index]:.2f})" if name != "Sconosciuto" else f"{name}"
        cv2.putText(frame, label, (left, top - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Mostra i dati della persona sotto il nome (una riga per ogni dato)
        if name != "Sconosciuto":
            y_offset = bottom + 20  # Partiamo da una posizione sotto il volto
            for line in person_info:
                cv2.putText(frame, line, (left, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                y_offset += 20  # Aggiungi un po' di spazio tra le righe

    # Mostra il frame
    cv2.imshow('Riconoscimento Facciale', frame)

    # Gestisci l'input da tastiera per cambiare la risoluzione
    key = cv2.waitKey(1) & 0xFF
    if key == ord('+'):
        window_width += 100  # Aumenta la larghezza
        window_height += 100  # Aumenta l'altezza
        cv2.resizeWindow("Riconoscimento Facciale", window_width, window_height)
    elif key == ord('-'):
        window_width -= 100  # Riduci la larghezza
        window_height -= 100  # Riduci l'altezza
        cv2.resizeWindow("Riconoscimento Facciale", window_width, window_height)

    # Premi 'q' per uscire
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
