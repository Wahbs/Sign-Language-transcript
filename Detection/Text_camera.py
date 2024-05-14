import cv2

# Fonction pour ajouter du texte en bas de l'ecran 
def add_text(frame, text):
    # Recuperer les dimensions du cadre
    height, width, _ = frame.shape
    # Définir la position du texte en bas
    bottom_left = (10, height - 10)

    #Definir la police, l'echelle, la couleur et l'epaisseur du texte
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (255, 255, 255)
    font_thickness = 2

    #Ajouter du texte au cadre
    cv2.putText(frame, text, bottom_left, font, font_scale, font_color, font_thickness)