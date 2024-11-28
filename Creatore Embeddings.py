import os
import numpy as np
from deepface import DeepFace

# **Fase 1: Creazione del database degli embedding**
# Percorso alla directory che contiene le cartelle di immagini degli utenti
users_dir = r"C:\Users\Gianni\Desktop\Foto_Riconoscimento"
if not os.path.exists(users_dir):
    raise FileNotFoundError(f"La directory {users_dir} non esiste!")

# Lista per salvare gli embedding
embedding_database = []

for user in os.listdir(users_dir):
    user_path = os.path.join(users_dir, user)
    if os.path.isdir(user_path):
        user_embeddings = []
        print(f"Processando immagini per l'utente: {user}")

        for img_file in os.listdir(user_path):
            img_path = os.path.join(user_path, img_file)
            try:
                # Calcola l'embedding con enforce_detection=False
                embedding = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                user_embeddings.append(np.array(embedding))
            except Exception as e:
                print(f"Errore nel processare {img_path}: {e}")

        # Calcola l'embedding medio per l'utente
        if user_embeddings:
            embedding_medio = np.mean(user_embeddings, axis=0)
            embedding_database.append({"name": user, "embedding": embedding_medio})

# Salva il database
np.save("embedding_database.npy", embedding_database)
print("Database degli embedding creato con successo!")
