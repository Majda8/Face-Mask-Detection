# Importer les packages nécessaires
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
    # Récupérer les dimensions du cadre et créer un blob à partir de celui-ci
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # Passer le blob à travers le réseau et obtenir les détections de visages
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # Initialiser les listes pour les visages détectés, leurs emplacements et les prédictions de notre réseau de détection de masque
    faces = []
    locs = []
    preds = []

    # Boucle sur les détections
    for i in range(0, detections.shape[2]):
        # Extraire la confiance (i.e., probabilité) associée à la détection
        confidence = detections[0, 0, i, 2]

        # Filtrer les détections faibles en s'assurant que la confiance est au-dessus du seuil minimum
        if confidence > 0.5:
            # Calculer les coordonnées (x, y) de la boîte englobante pour l'objet
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # S'assurer que les boîtes englobantes sont dans les dimensions du cadre
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Extraire la région d'intérêt du visage, convertir de BGR à RGB, redimensionner à 224x224 et prétraiter
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # Ajouter le visage et les boîtes englobantes à leurs listes respectives
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # Faire des prédictions seulement si au moins un visage a été détecté
    if len(faces) > 0:
        # Pour une inférence plus rapide, nous faisons des prédictions en lot sur *tous* les visages en même temps
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # Retourner un tuple des emplacements des visages et leurs prédictions correspondantes
    return (locs, preds)

# Charger notre modèle de détection de visage sérialisé depuis le disque
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Charger le modèle de détection de masque depuis le disque
maskNet = load_model("mask_detector_model.keras")

# Initialiser le flux vidéo
print("[INFO] démarrage du flux vidéo...")
vs = VideoStream(src=0).start()

# Boucle sur les cadres du flux vidéo
while True:
    # Récupérer le cadre du flux vidéo et le redimensionner à une largeur maximale de 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # Détecter les visages dans le cadre et déterminer s'ils portent un masque ou non
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # Boucle sur les emplacements des visages détectés et leurs prédictions correspondantes
    for (box, pred) in zip(locs, preds):
        # Décompresser la boîte englobante et les prédictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # Déterminer le label de classe et la couleur pour dessiner la boîte englobante et le texte
        label = "Masque" if mask > withoutMask else "Pas de masque"
        color = (0, 255, 0) if label == "Masque" else (0, 0, 255)

        # Inclure la probabilité dans le label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # Afficher le label et le rectangle de la boîte englobante sur le cadre de sortie
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Afficher le cadre de sortie
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Si la touche `q` est pressée, quitter la boucle
    if key == ord("q"):
        break

# Nettoyer un peu
cv2.destroyAllWindows()
vs.stop()
