import cv2
import face_recognition
import os
from screeninfo import get_monitors
import time
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import winsound
import geocoder

# Carica variabili dal file .env
load_dotenv()

# Ottieni credenziali email dal file .env
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# Controllo credenziali
if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
    raise ValueError("Le credenziali email non sono state trovate nel file .env!")

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

# Funzione per ottenere coordinate geografiche
def get_geographical_coordinates():
    try:
        location = geocoder.ip('me')
        if location.ok:
            return location.latlng  # Restituisce [latitudine, longitudine]
        else:
            return None
    except Exception as e:
        print(f"Errore durante il recupero delle coordinate geografiche: {e}")
        return None

# Funzione per inviare email con il link di Google Maps
def send_alert_email(subject, body):
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            msg = MIMEText(body)
            msg["From"] = EMAIL_ADDRESS
            msg["To"] = EMAIL_ADDRESS  # Invia a te stesso
            msg["Subject"] = subject
            server.sendmail(EMAIL_ADDRESS, EMAIL_ADDRESS, msg.as_string())
    except Exception as e:
        print(f"Errore durante l'invio dell'email: {e}")

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

        # Determina il colore del rettangolo in base allo status
        if "Pericoloso" in person_info:
            color = (0, 0, 255)  # Rosso per soggetti pericolosi
            label = f"{name} - PERICOLOSO"
            warning_message = f"Soggetto Pericoloso: {name} rilevato il {time.strftime('%Y-%m-%d %H:%M:%S')}"

            # Ottieni le coordinate geografiche
            coordinates = get_geographical_coordinates()
            if coordinates:
                latitude, longitude = coordinates
                warning_message += f" - Coordinate: Latitudine={latitude}, Longitudine={longitude}"
                google_maps_link = f"https://www.google.com/maps?q={latitude},{longitude}"
                warning_message += f"\nVisualizza su Google Maps: {google_maps_link}"

                print(f"Coordinate geografiche: Latitudine={latitude}, Longitudine={longitude}")
                print(f"Link Google Maps: {google_maps_link}")

            # Mostra il messaggio nella finestra
            cv2.putText(frame, warning_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            # Suona l'allarme
            winsound.Beep(1000, 1000)

            # Invia email con le coordinate e il link di Google Maps
            send_alert_email("ATTENZIONE: Soggetto Pericoloso Rilevato", warning_message)
        elif name == "Sconosciuto":
            color = (255, 255, 255)  # Bianco per soggetti sconosciuti
            label = "Sconosciuto"
        else:
            color = (0, 255, 0)  # Verde per soggetti normali (innocui)
            label = f"{name} ({distance[first_match_index]:.2f})"

        # Disegna il rettangolo attorno al volto
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Aggiungi il nome e la distanza
        cv2.putText(frame, label, (right + 10, top), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Stampa le informazioni aggiuntive
        y_offset = top
        for line in person_info:
            cv2.putText(frame, line, (right + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            y_offset += 20

    cv2.imshow('Riconoscimento Facciale', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
     
cap.release()
cv2.destroyAllWindows()
