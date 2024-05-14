import os
import subprocess
from django.contrib import messages
from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect, StreamingHttpResponse
from django.http import FileResponse
from django.http import HttpResponseNotFound
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import pickle
import pandas as pd
import numpy as np
#from .scripts_model import *
from .model import *
import cv2
from Detection.TensorflowP.scripts.train import Entraine_model

# Create your views here.

def accueil(request):

    return render(request, 'index.html')

def detecteur_objet(request):
    stop = 0
    transcript_file = 'media/transcript.txt'
    text_transcript = ''
    if request.method == 'GET':
        if 'y' in request.GET:
            if request.GET['y'] == 'stop':
                with open(transcript_file, 'r') as file:
                    text_transcript = file.read()
                stop = 1
                cv2.VideoCapture(0).release()

                return render(request, 'Detection/detecteur.html', {'stop':stop,
                                                                    'text_transcript': text_transcript})
        
    return render(request, 'Detection/detecteur.html', {'stop': stop})


def video_feed(request):
    return StreamingHttpResponse(main(), content_type='multipart/x-mixed-replace; boundary=frame')


def telecharge_transcription(request):
    file_path = 'media/transcript.txt'  # Remplacez cela par le chemin réel de votre fichier

    # Vérifier si le fichier existe
    if os.path.exists(file_path):
        try:
            response = FileResponse(open(file_path, 'rb'), content_type='text/plain')
            response['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
            return response
        except Exception as e:
            # Gérer les erreurs d'ouverture de fichier
            return HttpResponseNotFound("Erreur lors de l'ouverture du fichier.")
    else:
        return HttpResponseNotFound("Le fichier n'existe pas.")


def Ajout_geste(request):
    stop = 1
    labelimg = 'False'
    cv2.VideoCapture(0).release()
    if request.method == 'GET':
        if 'geste' in request.GET:
            new_geste = request.GET['geste']
            if new_geste != '' and new_geste != None:
                stop = 0
            return render(request, 'Detection/ajout_geste.html', {'stop': stop, 'geste':new_geste, 'labelimg':labelimg})
        
        if 'labelImg' in request.GET:
            try:
                labelimg = 'success'
                # Spécifiez le chemin complet de votre exécutable labelImg.py
                labelimg_path = 'Detection/TensorflowP/labelImg/labelImg.py'
                
                # Exécutez la commande pour lancer LabelImg
                subprocess.run(['python', labelimg_path])

                # Vous pouvez également spécifier d'autres paramètres de subprocess.run si nécessaire
                # Par exemple : subprocess.run(['python', labelimg_path, '--arg1', 'value1'])
                
                #return render(request, 'labelimg_launched.html', {'success': True})
            except Exception as e:
                labelimg = 'echec'
    if request.method == 'POST':
        proportion = request.POST['proportion']
        epochs = request.POST['epochs']
        batch_size = request.POST['batch_size']
        if 'use_tpu' in request.POST:
            use_tpu = True
        else:
            use_tpu = False
        #return HttpResponseRedirect(request.META.get('HTTP_REFERER'))
        #redirect('entrainement')
        context = {
            'proportion': proportion,
            'epochs': epochs,
            'batch_size': batch_size,
            'use_tpu': use_tpu,
            'labelimg': labelimg,
            'lien': 1,
        }
        return render(request, 'Detection/entrainement.html', context)
    return render(request, 'Detection/ajout_geste.html', {'stop': stop, 'labelimg':labelimg})

def video_ajout(request, new_geste):    
    return StreamingHttpResponse(Ajout_photo(new_geste), content_type='multipart/x-mixed-replace; boundary=frame')

# Dans votre fichier views.py
from django.http import JsonResponse
from django.views import View

class DecoupageView(View):
    def get(self, request):
        # Appel de la fonction decoupage avec une proportion de 70%
        # division des données
        progress_info = decoupage_upd(proportion_train=70, callback=self.update_progress)

        # Renvoie les informations sur l'état d'avancement au format JSON
        return JsonResponse(progress_info)

    def update_progress(self, progress_percent):
        # Cette fonction est appelée par la fonction decoupage pour mettre à jour la progression
        # Ici, vous pouvez stocker ou afficher la progression comme nécessaire
        print(f"Progress: {progress_percent}%")


def genere_tfrecord(request):
    try:
        # Exécutez le script pour les données d'entraînement
        train_script_command = f"python {SCRIPTS_PATH}/generate_tfrecord.py -x{IMAGE_PATH}/train -l {ANNOTATION_PATH}/label_map.pbtxt -o {ANNOTATION_PATH}/train.record -c {ANNOTATION_PATH}/train.csv "
        subprocess.run(train_script_command, shell=True, check=True)
        # Exécutez le script pour les données de test
        test_script_command = f"python {SCRIPTS_PATH}/generate_tfrecord.py -x{IMAGE_PATH}/test -l {ANNOTATION_PATH}/label_map.pbtxt -o {ANNOTATION_PATH}/test.record -c {ANNOTATION_PATH}/test.csv "
        subprocess.run(test_script_command, shell=True, check=True)

        return HttpResponse("Scripts exécutés avec succès.")

    except subprocess.CalledProcessError as e:
        return HttpResponse(f"Erreur lors de l'exécution des scripts : {str(e)}", status=500)

def configure_model(request, batch_size):
    ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
    NEW_CONFIG_PATH = MODEL_PATH+'/my_new_ssd_mobnet/pipeline.config'
    try:
        applique_configuration(batch_size=batch_size, CONFIG_PATH=NEW_CONFIG_PATH)
        return HttpResponse('configuré avec succès !')
    except Exception as e:
        return HttpResponse('erreur inconnue ')
    
def train_model(request, epochs, use_tpu):
    # Spécifiez le chemin complet vers votre script train.py
    script_path = 'Detection/TensorflowP/scripts/train1.py'
    WORKSPACE_PATH = 'Detection/TensorflowP/workspace'
    MODEL_PATH = WORKSPACE_PATH+'/models'
    PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
    NEW_CONFIG_PATH = MODEL_PATH+'/my_new_ssd_mobnet/pipeline.config'     # Pipeline du modele
    model_dir = MODEL_PATH+'/my_new_ssd_mobnet/checkpoint'           # dossier sauvegarde checkpoint du modele
    train_steps = epochs #1000
    use_tpu = True if use_tpu else False #False
    arguments = ['--pipeline_config_path', NEW_CONFIG_PATH, '--model_dir', model_dir,
                '--train_steps', train_steps, '--use_tpu', use_tpu,
            ]
    #train_script_command = f"python {SCRIPTS_PATH}/generate_tfrecord.py -x{IMAGE_PATH}/train -l {ANNOTATION_PATH}/label_map.pbtxt -o {ANNOTATION_PATH}/train.record -c {ANNOTATION_PATH}/train.csv "
    train_script_command =  f"python {script_path} "
    #result = subprocess.run(train_script_command, shell=True, check=True)
    # Exécutez le script en utilisant subprocess
    #train_script_command = f"python {script_path} --pipeline_config_path {CONFIG_PATH} --model_dir {model_dir} --train_steps {train_steps} --use_tpu {use_tpu}"
    #result = subprocess.run(train_script_command, shell=True, check=True)
    #result = subprocess.run(['python', script_path], capture_output=True, text=True)
    result = subprocess.run(['python', script_path] + arguments, capture_output=True, text=True)

    # Obtenez la sortie du script (result.stdout) et les erreurs éventuelles (result.stderr)
    output = result.stdout
    errors = result.stderr
    
    # Faites quelque chose avec la sortie ou les erreurs si nécessaire
    # Par exemple, renvoyez-les dans la réponse HTTP
    
    return HttpResponse(f"Output: {output}\nErrors: {errors}")
def update_progress(progress_percent):
    # Cette fonction est appelée par la fonction decoupage pour mettre à jour la progression
    # Ici, vous pouvez stocker ou afficher la progression comme nécessaire
    print(f"Progress: {progress_percent}%")

@csrf_exempt
def decoupage_donnees(request, proportion):
    # Vos procédures ici...
    # Exemple de mise à jour en temps réel
    message = "Découpage des données..."
    update = {"message": message, "progress": 10}
    try : 
        progress_info = decoupage_upd(proportion_train=proportion, callback=update_progress)
        return JsonResponse(progress_info)
    except Exception as e:
        update = {'erreur': e}
        # Envoi de la mise à jour au client
        return JsonResponse(update)

#@require_POST
@csrf_exempt
def entrainement_modele(request, proportion, epochs, batch_size, use_tpu=False, lien=1):
    try:
        print('decoupage.....')
        decoupage_donnees(request, proportion)
        print('generer tfrecord.....')
        genere_tfrecord(request)
        print('configuration du modèle.....')
        configure_model(request, batch_size)
        print('entraînement.....')
        #train_model(request, epochs, use_tpu)
        if lien == 1:
            Entraine_model(CONFIG_PATH = MODEL_PATH+'/my_new_ssd_mobnet/pipeline.config', model_dir = MODEL_PATH+'/my_new_ssd_mobnet/checkpoint', epochs=epochs, use_tpu=use_tpu)
        if lien == 2:
            test_script_command = f"python {SCRIPTS_PATH}/model_main_tf2.py --model_dir {MODEL_PATH+'/my_new_ssd_mobnet2'} --pipeline_config_path {MODEL_PATH+'/my_new_ssd_mobnet2/pipeline.config'} --num_train_steps {500}"
            subprocess.run(test_script_command, shell=True, check=True)
    except Exception as e :
        return HttpResponse(e)
    return JsonResponse({'succes': 'succes'})
    #return HttpResponseRedirect(request.META.get('HTTP_REFERER'))
    #return redirect('ajout_geste')
    """ try:
        # Démarrez le traitement ici
        messages.info(request, 'Début du découpage des données...')
        time.sleep(10)
        # Succès
        messages.success(request, 'Découpage des données terminé avec succès!')
        time.sleep(2)
    except Exception as e:
        # En cas d'erreur
        messages.error(request, f'Erreur lors du découpage des données: {str(e)}') """

def change_path(request, statut):
    if statut == 'old':
        old_path()
    if statut == 'new':
        update_path()
    if statut == 'restore':
        restore_path()
    return HttpResponseRedirect(request.META.get('HTTP_REFERER'))