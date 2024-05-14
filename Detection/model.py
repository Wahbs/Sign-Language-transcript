import os
import pandas as pd
from sklearn.model_selection import train_test_split
from .models import *
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.utils import visualization_utils as viz_utils
#from object_detection.utils import config_util
from object_detection.builders import model_builder
import cv2 
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

OLD_MODEL = Model_path.objects.filter(nom='old')[0].chemin
NEW_MODEL = Model_path.objects.filter(nom='new')[0].chemin
RESTORE_MODEL = Model_path.objects.filter(nom='restore')[0].chemin
ACTIVE_MODEL = Model_path.objects.filter(nom='active')[0].chemin

WORKSPACE_PATH = 'Detection/TensorflowP/workspace'
SCRIPTS_PATH = 'Detection/TensorflowP/scripts'
APIMODEL_PATH = 'Detection/TensorflowP/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models' 
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
#CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
#NEW_CONFIG_PATH = MODEL_PATH+'/my_new_ssd_mobnet/pipeline.config'
#CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/checkpoint'
#CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
CONFIG_PATH = ACTIVE_MODEL+'/pipeline.config'
NEW_CONFIG_PATH = NEW_MODEL+'/pipeline.config'
CHECKPOINT_PATH = ACTIVE_MODEL+'/checkpoint'
CUSTOM_MODEL_NAME = ACTIVE_MODEL
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

def recupere_index_ckpt(dossier):
    try:
        # Liste tous les fichiers dans le dossier
        fichiers = [f for f in os.listdir(dossier) if os.path.isfile(os.path.join(dossier, f))]

        # Filtrer les fichiers qui ont le format chk_{index}
        fichiers_chk = [f for f in fichiers if f.startswith('ckpt-')]

        if not fichiers_chk:
            raise ValueError("Aucun fichier ckpt_ trouvé dans le dossier.")

        # Extraire les index des fichiers
        indices = [int(f.split('-')[1].split('.')[0]) for f in fichiers_chk]

        # Trouver l'index maximal
        index_max = max(indices)

        # Construire le nom du fichier avec l'index maximal
        fichier_max = f"ckpt-{index_max}"
        print(fichier_max)
        chemin_fichier_max = os.path.join(dossier, fichier_max)

        return chemin_fichier_max

    except Exception as e:
        print(f"Une erreur s'est produite : {str(e)}")
        return None
    
def main(camera_active=True):
    WORKSPACE_PATH = 'Detection/TensorflowP/workspace'
    SCRIPTS_PATH = 'Detection/TensorflowP/scripts'
    APIMODEL_PATH = 'Detection/TensorflowP/models'
    ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
    IMAGE_PATH = WORKSPACE_PATH+'/images'
    MODEL_PATH = WORKSPACE_PATH+'/models'
    PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
    """ CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
    CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/checkpoint'
    CUSTOM_MODEL_NAME = 'my_ssd_mobnet' """
    #CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'
    if camera_active == False:
        cv2.VideoCapture(0).release()
        exit()
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    #ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-4')).expect_partial()
    ckpt.restore(recupere_index_ckpt(CHECKPOINT_PATH)).expect_partial()
    @tf.function
    def detect_fn(image):
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections

    category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')

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

    # Setup capture
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    texte = ''
    last_detected_label = ''
    # Vérifier si le fichier existe
    transcript_file = 'media/transcript.txt'
    
    if not os.path.exists(transcript_file):
        # Si le fichier n'existe pas, le créer et écrire une ligne d'en-tête
        with open(transcript_file, 'w') as file:
            file.write(f'{texte}')

    with open(transcript_file, 'w') as file:
        if camera_active == True:
            file.write(' ')
            while camera_active == True:
                ret, frame = cap.read()
                image_np = np.array(frame)

                #ancien = category_index[detect_class]['name']
                input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

                detections = detect_fn(input_tensor)

                num_detections = int(detections.pop('num_detections'))
                detections = {key: value[0, :num_detections].numpy()
                            for key, value in detections.items()}
                detections['num_detections'] = num_detections

                # detection_classes should be ints.
                detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

                label_id_offset = 1
                image_np_with_detections = image_np.copy()

                viz_utils.visualize_boxes_and_labels_on_image_array(
                            image_np_with_detections,
                            detections['detection_boxes'],
                            detections['detection_classes']+label_id_offset,
                            detections['detection_scores'],
                            category_index,
                            use_normalized_coordinates=True,
                            max_boxes_to_draw=1,
                            min_score_thresh=.8,
                            agnostic_mode=False)
                #add_text(frame, "yyuuy")
                detect_score = detections['detection_scores'][0] * 100
                detect_class = detections['detection_classes'][0] + 1
                detect_label = category_index[detect_class]['name']
                if detect_score > 80 and detect_label != last_detected_label:
                    texte += ' ' + detect_label
                    last_detected_label = detect_label
                    file.write(detect_label + ' ')
                    # Envoie le message au groupe WebSocket
                    #await send_detection_message_to_form(texte)
                    #await send_detection_message(texte)
                ali = ''
                tab = []
                tab1 = []
                tab = texte.split(' ')
                if len(tab)>10:
                    text_distinct = set(tab)
                    texte_distinct = list(text_distinct)
                    tab.clear()
                    texte = ''
                add_text(image_np_with_detections, texte)
                """ _, jpeg = cv2.imencode('.jpg', frame)
                frame = jpeg.tobytes()

                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n') """
                _, jpeg = cv2.imencode('.jpg', cv2.resize(image_np_with_detections, (640, 480)))
                frame = jpeg.tobytes()

                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                #cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))

                """ if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    break """

def main_modif(camera_active = True):
    WORKSPACE_PATH = 'Detection/TensorflowP/workspace'
    SCRIPTS_PATH = 'Detection/TensorflowP/scripts'
    APIMODEL_PATH = 'Detection/TensorflowP/models'
    ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
    IMAGE_PATH = WORKSPACE_PATH+'/images'
    MODEL_PATH = WORKSPACE_PATH+'/models'
    PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
    CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
    CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/checkpoint'
    CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
    if camera_active == False:
        cv2.VideoCapture(0).release()
        exit()
    
    #CONFIG_PATH = MODEL_PATH+'/'+CUSTOM_MODEL_NAME+'/pipeline.config'
    
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-4')).expect_partial()

    @tf.function
    def detect_fn(image):
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        print(image.shape)
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections

    category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')

    # Setup capture
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    texte = ''
    text = ''
    if camera_active == False:
        cap.release()

    # Vérifier si le fichier existe
    transcript_file = 'media/transcript.txt'
    
    if not os.path.exists(transcript_file):
        # Si le fichier n'existe pas, le créer et écrire une ligne d'en-tête
        with open(transcript_file, 'w') as file:
            file.write(' ')

    with open(transcript_file, 'w') as file:
        if camera_active == True:
            #file.write(' \n ')
            while camera_active == True:
                ret, frame = cap.read()
                image_np = np.array(frame)

                #ancien = category_index[detect_class]['name']
                input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

                detections = detect_fn(input_tensor)

                num_detections = int(detections.pop('num_detections'))
                detections = {key: value[0, :num_detections].numpy()
                            for key, value in detections.items()}
                detections['num_detections'] = num_detections

                # detection_classes should be ints.
                detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

                label_id_offset = 1
                image_np_with_detections = image_np.copy()

                viz_utils.visualize_boxes_and_labels_on_image_array(
                            image_np_with_detections,
                            detections['detection_boxes'],
                            detections['detection_classes']+label_id_offset,
                            detections['detection_scores'],
                            category_index,
                            use_normalized_coordinates=True,
                            max_boxes_to_draw=1,
                            min_score_thresh=.9,
                            agnostic_mode=False)
                #add_text(frame, "yyuuy")
                detect_score = detections['detection_scores'][0] * 100
                detect_class = detections['detection_classes'][0] + 1
                if detect_class == 12:
                    detect_label = 'toi'
                else:
                    detect_label = category_index[detect_class]['name']
                if detect_score > 90:
                    text += ' ' + detect_label
                    texte += ' ' + detect_label
                    file.write(detect_label + ' ')
                tab = text.split(' ')
                if len(tab)>10:
                    tab.clear()
                    text = ''
                add_text(image_np_with_detections, text)
                """ _, jpeg = cv2.imencode('.jpg', frame)
                frame = jpeg.tobytes()

                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n') """
                _, jpeg = cv2.imencode('.jpg', cv2.resize(image_np_with_detections, (640, 480)))
                frame = jpeg.tobytes()

                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))

def Ajout_photo(new_geste):
    IMAGES_PATH = 'Detection/TensorflowP/workspace/images/collectedimages'
    #labels = ['hello', 'thanks', 'yes', 'no', 'iloveyou']
    labels = []
    labels.append(new_geste)
    number_imgs = 10

    for label in labels:
        # !mkdir {'Tensorflow\workspace\images\collectedimages\\'+label}
        IMAGE_PATH1 = IMAGES_PATH+'/'+label
        
        # Vérifier si le dossier existe, sinon le créer
        if not os.path.exists(IMAGE_PATH1):
            os.makedirs(IMAGE_PATH1)
        
        texte = 'Recuperation des images pour {}'.format(label)
        print('Recuperation des images pour {}'.format(label))
        time.sleep(5)
        
        # Ouvrir la capture vidéo à partir de la caméra
        cap = cv2.VideoCapture(0)

        for imgnum in range(number_imgs):
            # Lire une image de la capture video
            ret, frame = cap.read()
            time.sleep(3)
            # Afficher l'image dans une fenêtre
            # imgname = os.path.join(IMAGES_PATH, label, label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
            imgname = label+'_'+'{}.jpg'.format(imgnum)
            
            # Chemin complet de l'image
            image_path = os.path.join(IMAGE_PATH1, imgname)
            cv2.imwrite(image_path, frame)
            texte = f"Image enregistrée : {imgname}"
            if imgnum == number_imgs -1 :
                add_text(frame, 'Enregistrement terminé')
            else:
                add_text(frame, texte)
            #cv2.imshow('frame', frame)
            _, jpeg = cv2.imencode('.jpg', cv2.resize(frame, (640, 480)))
            frame_transform = jpeg.tobytes()

            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_transform + b'\r\n\r\n')
            
            
            """ if cv2.waitKey(1) & 0xFF == ord('q'):
                break """

        cap.release()


import os
import random
import shutil

def decoupage(proportion_train):
    # Chemin du dossier contenant les images labelisées
    dossier_images = 'Detection/TensorflowP/workspace/images/all'

    # Chemin des dossiers de sortie train et test
    dossier_train = 'Detection/TensorflowP/workspace/images/train'
    dossier_test = 'Detection/TensorflowP/workspace/images/test'

    # Création des dossiers de sortie s'ils n'existent pas
    os.makedirs(dossier_train, exist_ok=True)
    os.makedirs(dossier_test, exist_ok=True)

    # Liste des fichiers image et XML
    fichiers = [f for f in os.listdir(dossier_images) if f.endswith(('.jpg', '.xml'))]

    # Dictionnaire pour stocker les couples image-label par classe
    couples_par_classe = {}

    for fichier in fichiers:
        # Exemple de nom de fichier : hello_1.png
        nom_classe, _ = os.path.splitext(fichier.split('_' or '.')[0])

        # Ajout du couple image-label à la liste correspondante dans le dictionnaire
        if nom_classe not in couples_par_classe:
            couples_par_classe[nom_classe] = []
        couples_par_classe[nom_classe].append(fichier)

    for classe, couples in couples_par_classe.items():
        # Calcul du nombre d'images pour le dossier de test
        nb_images_test = int(len(couples) * (1 - proportion_train/100))

        # Sélection aléatoire des couples pour le dossier de test
        couples_test = random.sample(couples, nb_images_test)

        # Déplacement des couples vers les dossiers train et test
        for couple in couples:
            chemin_source = os.path.join(dossier_images, couple)
            if couple in couples_test:
                chemin_destination = os.path.join(dossier_test, couple)
            else:
                chemin_destination = os.path.join(dossier_train, couple)
            shutil.move(chemin_source, chemin_destination)

# Utilisation de la fonction avec une proportion de 70% pour le dossier train
# decoupage(proportion_train=70)
# Dans votre fichier decoupage.py
import os
import shutil

def decoupage_upd(proportion_train=70, callback=None):
    # Obtenir la liste des fichiers dans le dossier 'all'
    # Chemin du dossier contenant les images labelisées
    dossier_images = 'Detection/TensorflowP/workspace/images/all'

    # Chemin des dossiers de sortie train et test
    dossier_train = 'Detection/TensorflowP/workspace/images/train'
    dossier_test = 'Detection/TensorflowP/workspace/images/test'

    # Liste des fichiers image et XML
    fichiers = [f for f in os.listdir(dossier_images) if f.endswith(('.jpg'))]
    nom_classes = []
    dict = {}
    donnees = pd.DataFrame(columns = ['classe', 'fichier'])
    for fichier in fichiers:
        classe = ''
        for l in fichier:
            if l == '.' or l == '_':
                break
            classe += l 
        if classe not in nom_classes:
            nom_classes.append(classe)
        #donnees['classe'] = classe
        #donnees['fichier'] = fichier
        donnees = donnees.append({'classe': classe, 'fichier': fichier.replace('.jpg', '')}, ignore_index=True)
    print(donnees)

# Supposons que df soit votre DataFrame avec les colonnes 'Classe' et 'Fichier'
# Remplacez 'VotreDataFrame' par le nom de votre DataFrame si nécessaire

    # Grouper par classe
    grouped_df = donnees.groupby('classe')

    # Initialiser les DataFrames pour les ensembles de formation et de test
    train_df = pd.DataFrame(columns=['classe', 'fichier'])
    test_df = pd.DataFrame(columns=['classe', 'fichier'])

    # Pour chaque groupe (classe), divisez les données
    for classe, group in grouped_df:
        # Divisez le groupe en ensembles de formation et de test (70% train, 30% test)
        rate = round((100 - proportion_train)/100, 1)
        train_group, test_group = train_test_split(group, test_size=rate, random_state=42)

        # Ajoutez les ensembles de formation et de test aux DataFrames respectifs
        train_df = train_df.append(train_group, ignore_index=True)
        test_df = test_df.append(test_group, ignore_index=True)
    #print(nom_classes)
    print(train_df.duplicated().sum())
    print(train_df.duplicated().sum())
    train_df = train_df.drop_duplicates(subset='fichier')
    test_df = test_df.drop_duplicates(subset='fichier')
    print("Ensemble de formation :\n", train_df)
    print("\nEnsemble de test :\n", test_df)

    files = os.listdir(dossier_images)
    
    # Calculer le nombre de fichiers pour la proportion spécifiée
    num_files = len(files)
    num_train = int(num_files * proportion_train / 100)
    
    # Création des dossiers de sortie s'ils n'existent pas
    os.makedirs(dossier_train, exist_ok=True)
    os.makedirs(dossier_test, exist_ok=True)
    
    test_df['type'] = 'test'
    train_df['type'] = 'train'

    df = pd.concat([test_df, train_df], ignore_index=True)
    print(df)
    # Copier les fichiers dans les dossiers 'train' et 'test'
    for i, row in df.iterrows():
        file_xml = row['fichier'] + '.xml'
        file_jpg = row['fichier'] + '.jpg'
        src_path_xml = os.path.join(dossier_images, file_xml)
        src_path_jpg = os.path.join(dossier_images, file_jpg)
        if row['type'] == 'test':
            dest_folder = dossier_test
        elif row['type'] == 'train':
            dest_folder = dossier_train
        dest_path_xml = os.path.join(dest_folder, file_xml)
        dest_path_jpg = os.path.join(dest_folder, file_jpg)
        shutil.copy(src_path_xml, dest_path_xml)
        shutil.copy(src_path_jpg, dest_path_jpg)
        
        # Calculer et appeler la progression
        progress_percent = int((i + 1) / num_files * 100) * 2
        if callback:
            callback(progress_percent)
    cree_label_map()
    # Renvoyer des informations supplémentaires si nécessaire
    progress_info = {'status': 'division et labelisation effectuées avec succès'}
    return progress_info


def cree_label_map():
    train_folder_path = 'Detection/TensorflowP/workspace/images/train'

    labels = []

    # Obtenez tous les fichiers dans le dossier train
    files = os.listdir(train_folder_path)

    # Filtrer uniquement les fichiers avec l'extension .xml
    xml_files = [file for file in files if file.endswith('.xml')]

    # Parcourez les fichiers XML et extrayez les noms de classe
    for xml_file in xml_files:
        xml_path = os.path.join(train_folder_path, xml_file)

        with open(xml_path, 'r') as f:
            content = f.read()

            # Recherchez la balise <name> pour extraire le nom de la classe
            start = content.find('<name>') + len('<name>')
            end = content.find('</name>', start)
            class_name = content[start:end].strip()

            # Vérifiez si la classe existe déjà dans la liste
            if not any(label['name'] == class_name for label in labels):
                labels.append({'name': class_name, 'id': len(labels) + 1})

    # Écrivez la liste des labels dans le fichier label_map.pbtxt
    with open(ANNOTATION_PATH + '/label_map.pbtxt', 'w') as f:
        for label in labels:
            f.write('item { \n')
            f.write('\tname:\"{}\"\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')

def get_num_classes_from_label_map(label_map_path):
    with open(label_map_path, 'r') as file:
        lines = file.readlines()

    num_classes = 0
    for line in lines:
        if 'id' in line:
            num_classes += 1

    return num_classes

def applique_configuration(batch_size = 4, CONFIG_PATH=NEW_CONFIG_PATH, ANNOTATION_PATH=ANNOTATION_PATH):
    config = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
    label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
    num_classes = get_num_classes_from_label_map(label_map_path)

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.io.gfile.GFile(CONFIG_PATH, "r") as f:                                                                                                                                                                                                                     
        proto_str = f.read()                                                                                                                                                                                                                                          
        text_format.Merge(proto_str, pipeline_config)  

    pipeline_config.model.ssd.num_classes = num_classes
    pipeline_config.train_config.batch_size = batch_size
    pipeline_config.train_config.fine_tune_checkpoint = PRETRAINED_MODEL_PATH+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint/ckpt-0'
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_input_reader.label_map_path= ANNOTATION_PATH + '/label_map.pbtxt'
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/train.record']
    pipeline_config.eval_input_reader[0].label_map_path = ANNOTATION_PATH + '/label_map.pbtxt'
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [ANNOTATION_PATH + '/test.record']

    config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
    with tf.io.gfile.GFile(CONFIG_PATH, "wb") as f:                                                                                                                                                                                                                     
        f.write(config_text) 

def permuter_dossiers(dossier_source, dossier_destination):
    # Créer des noms temporaires pour les dossiers
    src = dossier_source
    dst = dossier_destination
    dossier_temporaire_source = dossier_source + "_temp"
    dossier_temporaire_destination = dossier_destination + "_temp"

    try:
        # Renommer les dossiers d'origine
        os.rename(dossier_source, dossier_temporaire_source)
        os.rename(dossier_destination, dossier_temporaire_destination)

        # Renommer les dossiers temporaires avec les noms originaux
        os.rename(dossier_temporaire_source, dst)
        os.rename(dossier_temporaire_destination, src)

        print(f"Les dossiers {dossier_source} et {dossier_destination} ont été permutés avec succès.")
    except Exception as e:
        print(f"Une erreur s'est produite : {str(e)}")

def update_path():
    fichier_config = PRETRAINED_MODEL_PATH+'/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config'
    model_active = Model_path.objects.filter(nom='active')[0]
    if os.listdir(NEW_MODEL).count('checkpoint') > 0:
        model_active.chemin = Model_path.objects.filter(nom='active1')[0].chemin
        model_active.save()
        permuter_dossiers(OLD_MODEL, model_active)
        permuter_dossiers(model_active, NEW_MODEL)
        """ Model_path.objects.filter(nom='active')[0].chemin = ACTIVE_MODEL
        Model_path.objects.filter(nom='old')[0].chemin = OLD_MODEL """
        # Supprimer le dossier et son contenu
        shutil.rmtree(NEW_MODEL)
        # Recréer le dossier vide
        os.makedirs(NEW_MODEL)
        shutil.copy2(fichier_config, NEW_MODEL)

def restore_path():
    model_active = Model_path.objects.filter(nom='active')[0]
    model_restore = Model_path.objects.filter(nom='restore')[0]
    model_active.chemin = model_restore.chemin
    model_active.save()

def old_path():
    model_active = Model_path.objects.filter(nom='active')[0].chemin
    model_old = Model_path.objects.filter(nom='old')[0].chemin
    for f in os.listdir(model_old):
        shutil.copy2(model_old+'/'+f, model_active)
    """  model_active.chemin = model_old.chemin
    model_old.save() """