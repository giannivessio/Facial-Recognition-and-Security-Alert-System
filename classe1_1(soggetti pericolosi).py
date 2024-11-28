import cv2
import face_recognition
import os
from screeninfo import get_monitors
import time
import winsound  # Importa winsound per suonare un allarme (solo su Windows)

# Ottieni la risoluzione dello schermo
monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

# Percorso della cartella principale che contiene le sottocartelle con le immagini
main_directory = r"C:\Users\Gianni\Desktop\Foto_Riconoscimento"

# Liste per memorizzare gli encoding dei volti, i nomi associati e i dati
known_face_encodings = []
known_face_names = []
person_data = {}

# Esplora tutte le sottocartelle nella cartella principale
for person_name in os.listdir(main_directory):
    person_folder = os.path.join(main_directory, person_name)
    
    if os.path.isdir(person_folder):
        # Aggiungi i dati della persona leggendo il file .txt
        data_file = os.path.join(person_folder, f"Dati {person_name}.txt")
        if os.path.isfile(data_file):
            with open(data_file, 'r') as file:
                person_info = file.read().splitlines()
            person_data[person_name] = person_info

            # Verifica se il soggetto è pericoloso
            for line in person_info:
                if "Status: Pericoloso" in line:
                    person_data[person_name].append("Pericoloso")
        else:
            person_data[person_name] = ["Dati non disponibili"]

        # Esplora tutte le immagini nella cartella della persona
        for image_file in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_file)
            
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    volto_image = face_recognition.load_image_file(image_path)
                    volto_encoding = face_recognition.face_encodings(volto_image)
                    
                    if volto_encoding:
                        known_face_encodings.append(volto_encoding[0])
                        known_face_names.append(person_name)
                except Exception as e:
                    print(f"Errore nel caricare l'immagine {image_path}: {e}")

# Impostazione iniziale della finestra della videocamera
window_width = 1280
window_height = 720

# Inizializza la videocamera
cap = cv2.VideoCapture(0)

# Soglia di precisione per il riconoscimento
precision_threshold = 0.55

# Imposta la risoluzione iniziale della videocamera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)

cv2.namedWindow("Riconoscimento Facciale", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Riconoscimento Facciale", window_width, window_height)

frame_counter = 0  # Variabile per contare i fotogrammi

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Elabora ogni 5 fotogrammi (riduce il carico di lavoro)
    frame_counter += 1
    if frame_counter % 5 != 0:
        continue

    # Ridurre la risoluzione del frame per migliorare le performance
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Calcola il fattore di scala
    scale_x = frame.shape[1] / small_frame.shape[1]
    scale_y = frame.shape[0] / small_frame.shape[0]

    # Converti il frame in RGB per il riconoscimento facciale
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Usa il modello HOG per il rilevamento dei volti, più veloce (meno preciso)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        distance = face_recognition.face_distance(known_face_encodings, face_encoding)

        name = "Sconosciuto"
        person_info = []
        if True in matches:
            first_match_index = matches.index(True)
            if distance[first_match_index] < precision_threshold:
                name = known_face_names[first_match_index]
                person_info = person_data.get(name, ["Dati non disponibili"])

        # Ridimensiona le coordinate del volto dalla versione ridotta alla versione originale
        top, right, bottom, left = face_location
        left = int(left * scale_x)
        right = int(right * scale_x)
        top = int(top * scale_y)
        bottom = int(bottom * scale_y)

        margin = 0.2
        width = right - left
        height = bottom - top
        left = int(left - margin * width)
        right = int(right + margin * width)
        top = int(top - margin * height)
        bottom = int(bottom + margin * height)

        left = max(left, 0)
        right = min(right, frame.shape[1])
        top = max(top, 0)
        bottom = min(bottom, frame.shape[0])

        # Determina il colore del rettangolo in base allo status
        if "Pericoloso" in person_info:
            color = (0, 0, 255)  # Rosso per soggetti pericolosi
            label = f"{name} - PERICOLOSO"
            warning_message = "!!! ATTENZIONE: Soggetto Pericoloso Rilevato !!!"
            cv2.putText(frame, warning_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            # Esegui il suono di allarme (suona per 1 secondo)
            winsound.Beep(1000, 1000)  # Frequenza 1000 Hz, durata 1000 ms (1 secondo)

            # Salva un log (scrittura su file ogni 10 rilevamenti)
            if frame_counter % 50 == 0:
                with open("log_soggetti_pericolosi.txt", "a") as log_file:
                    log_file.write(f"{name} rilevato il {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        else:
            color = (0, 255, 0)  # Verde per soggetti normali
            label = f"{name} ({distance[first_match_index]:.2f})"

        # Disegna il rettangolo attorno al volto
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Posiziona la descrizione a destra del riquadro
        x_offset = right + 10  # Aggiungi un margine di 10 pixel a destra
        y_offset = top + 20  # Partiamo dal top del volto per la descrizione
        for line in person_info:
            cv2.putText(frame, line, (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            y_offset += 20  # Incrementa la posizione verticale per la descrizione successiva

        # Metti il nome sopra il riquadro a destra
        cv2.putText(frame, label, (x_offset, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Riconoscimento Facciale', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('+'):
        window_width += 100
        window_height += 100
        cv2.resizeWindow("Riconoscimento Facciale", window_width, window_height)
    elif key == ord('-'):
        window_width -= 100
        window_height -= 100
        cv2.resizeWindow("Riconoscimento Facciale", window_width, window_height)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
