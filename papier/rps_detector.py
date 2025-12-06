# Installation des packages nécessaires (à exécuter dans le terminal PyCharm)
# pip install ultralytics opencv-python pillow matplotlib pyyaml

from ultralytics import YOLO
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import yaml
import os
import pathlib
import torch

# FIX pour Windows: Convertir PosixPath en WindowsPath
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# 1. Charger le fichier data.yaml pour voir les classes
data_yaml_path = r'C:\Users\mouma\AppData\Roaming\JetBrains\PyCharm2025.1\scratches\papier\data.yaml'
with open(data_yaml_path, 'r') as file:
    data_config = yaml.safe_load(file)

print("Classes détectables:")
for i, name in enumerate(data_config['names']):
    print(f"{i}: {name}")
print(f"\nNombre total de classes: {data_config['nc']}")

# 2. Charger le modèle entraîné
model_path = r'C:\Users\mouma\AppData\Roaming\JetBrains\PyCharm2025.1\scratches\papier\best.pt'
try:
    model = YOLO(model_path)
    print("\nModèle chargé avec succès!")
except Exception as e:
    print(f"\nErreur lors du chargement du modèle: {e}")
    print("\nEssai de rechargement avec correction...")
    ckpt = torch.load(model_path, map_location='cpu')
    model = YOLO('yolov8n.pt')
    model.model.load_state_dict(ckpt['model'].state_dict())
    print("Modèle chargé avec succès après correction!")


# ============================================
# FONCTIONS DE DÉTECTION
# ============================================

def detect_on_image(image_path):
    """Détection sur une seule image"""
    if not os.path.exists(image_path):
        print(f"❌ L'image '{image_path}' n'existe pas!")
        return None

    results = model(image_path)
    annotated_image = results[0].plot()
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image_rgb)
    plt.axis('off')
    plt.title('Détection Pierre-Papier-Ciseaux')
    plt.show()

    print("\n=== Détections ===")
    for i, box in enumerate(results[0].boxes):
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = data_config['names'][class_id]
        print(f"Geste {i + 1}: {class_name} - Confiance: {confidence:.2%}")

    return results


def detect_on_folder(folder_path, save_dir='detections'):
    """Détection sur toutes les images d'un dossier"""
    if not os.path.exists(folder_path):
        print(f"❌ Le dossier '{folder_path}' n'existe pas!")
        return None

    os.makedirs(save_dir, exist_ok=True)
    results = model.predict(
        source=folder_path,
        save=True,
        save_txt=True,
        conf=0.25,
        project=save_dir,
        name='predictions'
    )

    print(f"\n✅ Détections sauvegardées dans: {save_dir}/predictions")
    print(f"📊 Nombre d'images traitées: {len(results)}")
    return results


def detect_from_url(image_url):
    """Détection depuis une URL"""
    try:
        results = model(image_url)
        annotated_image = results[0].plot()
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 8))
        plt.imshow(annotated_image_rgb)
        plt.axis('off')
        plt.title('Détection Pierre-Papier-Ciseaux')
        plt.show()

        return results
    except Exception as e:
        print(f"❌ Erreur lors du chargement de l'URL: {e}")
        return None


def detect_on_video(video_path):
    """Détection sur une vidéo"""
    if not os.path.exists(video_path):
        print(f"❌ La vidéo '{video_path}' n'existe pas!")
        return None

    results = model.predict(
        source=video_path,
        save=True,
        conf=0.25,
        project='video_detections',
        name='output'
    )

    print(f"✅ Vidéo sauvegardée dans: video_detections/output")
    return results


def detect_custom(image_path, conf_threshold=0.25, iou_threshold=0.45):
    """Détection avec paramètres personnalisés"""
    if not os.path.exists(image_path):
        print(f"❌ L'image '{image_path}' n'existe pas!")
        return None

    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        save=True,
        project='custom_detections',
        name='results'
    )

    annotated_image = results[0].plot()
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image_rgb)
    plt.axis('off')
    plt.title(f'Détection (conf={conf_threshold}, iou={iou_threshold})')
    plt.show()

    return results


def detect_webcam():
    """Détection en temps réel sur webcam"""
    cap = cv2.VideoCapture('http://192.168.137.80:4747/video')

    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la webcam!")
        return

    print("🎥 Webcam activée. Appuyez sur 'q' pour quitter")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.25)
        annotated_frame = results[0].plot()

        # Afficher les détections sur la frame
        cv2.imshow('Détection Pierre-Papier-Ciseaux - Appuyez sur Q pour quitter', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Webcam fermée")


def play_game_mode():
    """Mode jeu: jouer contre l'ordinateur"""
    import random

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Impossible d'ouvrir la webcam!")
        return

    choices = ['Paper', 'Rock', 'Scissors']
    player_score = 0
    computer_score = 0

    print("\n🎮 MODE JEU PIERRE-PAPIER-CISEAUX")
    print("=" * 50)
    print("📹 Montrez votre geste à la caméra!")
    print("🖐️  Appuyez sur ESPACE pour valider votre choix")
    print("⏹️  Appuyez sur Q pour quitter")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.4)
        annotated_frame = results[0].plot()

        # Afficher le score
        cv2.putText(annotated_frame, f"Joueur: {player_score} | Ordinateur: {computer_score}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, "ESPACE = Jouer | Q = Quitter",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Jeu Pierre-Papier-Ciseaux', annotated_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):  # Espace
            # Détecter le geste du joueur
            if len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                class_id = int(box.cls[0])
                player_choice = data_config['names'][class_id]

                # Choix de l'ordinateur
                computer_choice = random.choice(choices)

                # Déterminer le gagnant
                print(f"\n🎯 Joueur: {player_choice}")
                print(f"🤖 Ordinateur: {computer_choice}")

                if player_choice == computer_choice:
                    print("⚖️  Égalité!")
                elif (player_choice == 'Rock' and computer_choice == 'Scissors') or \
                        (player_choice == 'Paper' and computer_choice == 'Rock') or \
                        (player_choice == 'Scissors' and computer_choice == 'Paper'):
                    print("🎉 Vous gagnez!")
                    player_score += 1
                else:
                    print("😞 L'ordinateur gagne!")
                    computer_score += 1
            else:
                print("❌ Aucun geste détecté!")

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 50)
    print(f"🏆 SCORE FINAL")
    print(f"Joueur: {player_score} | Ordinateur: {computer_score}")
    if player_score > computer_score:
        print("🎉 Félicitations! Vous avez gagné!")
    elif player_score < computer_score:
        print("😞 L'ordinateur a gagné!")
    else:
        print("⚖️  Match nul!")
    print("=" * 50)


# ============================================
# MENU INTERACTIF
# ============================================

def afficher_menu():
    """Affiche le menu principal"""
    print("\n" + "=" * 60)
    print("✊✋✌️  SYSTÈME DE DÉTECTION PIERRE-PAPIER-CISEAUX ✊✋✌️")
    print("=" * 60)
    print("\n📋 OPTIONS DISPONIBLES:\n")
    print("  1️⃣  - Détecter sur une image")
    print("  2️⃣  - Détecter sur un dossier d'images")
    print("  3️⃣  - Détecter depuis une URL")
    print("  4️⃣  - Détecter sur une vidéo")
    print("  5️⃣  - Détection personnalisée (seuils ajustables)")
    print("  6️⃣  - Détection webcam en temps réel")
    print("  7️⃣  - Détecter sur le dossier test complet")
    print("  8️⃣  - 🎮 MODE JEU: Jouer contre l'ordinateur!")
    print("  0️⃣  - Quitter le programme")
    print("\n" + "=" * 60)


def option_1():
    """Option 1: Détection sur une image"""
    print("\n🖼️  DÉTECTION SUR UNE IMAGE")
    print("-" * 40)
    chemin = input("📁 Entrez le chemin de l'image: ").strip()
    if chemin:
        detect_on_image(chemin)
    else:
        print("❌ Chemin invalide!")


def option_2():
    """Option 2: Détection sur un dossier"""
    print("\n📂 DÉTECTION SUR UN DOSSIER")
    print("-" * 40)
    chemin = input("📁 Entrez le chemin du dossier: ").strip()
    save_dir = input("💾 Dossier de sauvegarde (appuyez sur Entrée pour 'detections'): ").strip()

    if not save_dir:
        save_dir = 'detections'

    if chemin:
        detect_on_folder(chemin, save_dir)
    else:
        print("❌ Chemin invalide!")


def option_3():
    """Option 3: Détection depuis URL"""
    print("\n🌐 DÉTECTION DEPUIS UNE URL")
    print("-" * 40)
    url = input("🔗 Entrez l'URL de l'image: ").strip()
    if url:
        detect_from_url(url)
    else:
        print("❌ URL invalide!")


def option_4():
    """Option 4: Détection sur vidéo"""
    print("\n🎬 DÉTECTION SUR UNE VIDÉO")
    print("-" * 40)
    chemin = input("📁 Entrez le chemin de la vidéo: ").strip()
    if chemin:
        detect_on_video(chemin)
    else:
        print("❌ Chemin invalide!")


def option_5():
    """Option 5: Détection personnalisée"""
    print("\n⚙️  DÉTECTION PERSONNALISÉE")
    print("-" * 40)
    chemin = input("📁 Entrez le chemin de l'image: ").strip()

    if not chemin:
        print("❌ Chemin invalide!")
        return

    try:
        conf = input("🎯 Seuil de confiance (0-1, défaut=0.25): ").strip()
        conf = float(conf) if conf else 0.25

        iou = input("🔄 Seuil IoU (0-1, défaut=0.45): ").strip()
        iou = float(iou) if iou else 0.45

        detect_custom(chemin, conf, iou)
    except ValueError:
        print("❌ Valeurs invalides! Utilisation des valeurs par défaut.")
        detect_custom(chemin)


def option_6():
    """Option 6: Webcam"""
    print("\n🎥 DÉTECTION WEBCAM EN TEMPS RÉEL")
    print("-" * 40)
    print("⚠️  La webcam va s'ouvrir. Appuyez sur 'Q' pour arrêter.")
    input("Appuyez sur Entrée pour continuer...")
    detect_webcam()


def option_7():
    """Option 7: Dossier test complet"""
    print("\n🧪 DÉTECTION SUR LE DOSSIER TEST")
    print("-" * 40)

    test_folder = '../test/images'
    chemin = input(f"📁 Chemin (défaut='{test_folder}'): ").strip()

    if not chemin:
        chemin = test_folder

    if not os.path.exists(chemin):
        print(f"❌ Le dossier '{chemin}' n'existe pas!")
        return

    results = model.predict(
        source=chemin,
        save=True,
        conf=0.25,
        project='rps_detections',
        name='test_results',
        save_txt=True
    )

    print(f"\n✅ Détections terminées!")
    print(f"💾 Résultats sauvegardés dans: rps_detections/test_results")
    print(f"📊 Nombre d'images traitées: {len(results)}")

    # Afficher quelques exemples
    result_path = 'rps_detections/test_results'
    if os.path.exists(result_path):
        result_images = [f for f in os.listdir(result_path) if f.endswith(('.jpg', '.png'))]

        if result_images:
            afficher = input(f"\n🖼️  Afficher les 3 premières détections? (o/n): ").strip().lower()
            if afficher == 'o':
                for i, img_name in enumerate(result_images[:3]):
                    img_path = os.path.join(result_path, img_name)
                    img = Image.open(img_path)

                    plt.figure(figsize=(10, 7))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(f'Détection {i + 1}: {img_name}')
                    plt.show()


def option_8():
    """Option 8: Mode jeu"""
    play_game_mode()


def menu_principal():
    """Boucle principale du menu"""
    while True:
        afficher_menu()
        choix = input("👉 Choisissez une option (0-8): ").strip()

        if choix == '1':
            option_1()
        elif choix == '2':
            option_2()
        elif choix == '3':
            option_3()
        elif choix == '4':
            option_4()
        elif choix == '5':
            option_5()
        elif choix == '6':
            option_6()
        elif choix == '7':
            option_7()
        elif choix == '8':
            option_8()
        elif choix == '0':
            print("\n👋 Au revoir! Merci d'avoir utilisé le système de détection.")
            break
        else:
            print("\n❌ Option invalide! Veuillez choisir entre 0 et 8.")

        input("\n⏸️  Appuyez sur Entrée pour continuer...")


# ============================================
# POINT D'ENTRÉE
# ============================================

if __name__ == "__main__":
    menu_principal()