import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection import model_lib_v2

def main(_):
    WORKSPACE_PATH = 'Detection/TensorflowP/workspace'
    MODEL_PATH = WORKSPACE_PATH+'/models'
    PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
    CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'

    # Spécifiez le chemin vers le fichier de configuration du modèle
    pipeline_config_path = CONFIG_PATH

    # Spécifiez le chemin vers le dossier de sauvegarde des checkpoints du modèle
    model_dir = MODEL_PATH+'/my_ssd_mobnet/checkpoint'

    # Chargez la configuration du modèle
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    model_config = configs['model']

    # Créez une configuration d'entraînement
    train_config = configs['train_config']

    # Créez une configuration d'évaluation (facultatif)
    eval_config = configs['eval_config']

    # Démarrez l'entraînement
    model_lib_v2.train_loop(
        pipeline_config_path=CONFIG_PATH,
        pipeline_config=model_config,
        train_config=train_config,
        model_dir=model_dir,
        config_override=None,
        train_steps=1000,
        use_tpu=False,
        save_final_config=True,
        eval_train_data=False,
        eval_config=eval_config
    )

if __name__ == '__main__':
    # Lancez l'exécution du script
    tf.compat.v1.app.run()
