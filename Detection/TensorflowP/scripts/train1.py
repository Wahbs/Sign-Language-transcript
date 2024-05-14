import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection import model_lib_v2
import argparse

def main(_):
    script_path = 'Detection/TensorflowP/scripts/train1.py'
    WORKSPACE_PATH = 'Detection/TensorflowP/workspace'
    MODEL_PATH = WORKSPACE_PATH+'/models'
    PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
    CONFIG_PATH = MODEL_PATH+'/my_new_ssd_mobnet/pipeline.config'     # Pipeline du modele
    model_dir = MODEL_PATH+'/my_new_ssd_mobnet/checkpoint' 
    parser = argparse.ArgumentParser(description='Train an object detection model.')
    parser.add_argument('--pipeline_config_path', type=str, default=CONFIG_PATH, help='Path to pipeline config file.')
    parser.add_argument('--model_dir', type=str, default=model_dir, help='Path to output model directory.')
    parser.add_argument('--train_steps', type=int, default=1000, help='Number of train steps.')
    parser.add_argument('--use_tpu', action='store_true', help='Whether the job is executing on a TPU.')
    parser.add_argument('--eval_train_data', action='store_true', help='Enable evaluating on train data.')

    args = parser.parse_args()

    if not args.pipeline_config_path or not args.model_dir:
        parser.error('Both pipeline_config_path and model_dir are required.')

    # Chargez la configuration du modèle
    configs = config_util.get_configs_from_pipeline_file(args.pipeline_config_path)
    model_config = configs['model']

    # Créez une configuration d'entraînement
    train_config = configs['train_config']

    # Créez une configuration d'évaluation (facultatif)
    eval_config = configs['eval_config']

    # Démarrez l'entraînement
    model_lib_v2.train_loop(
        pipeline_config_path=args.pipeline_config_path,
        pipeline_config=model_config,
        train_config=train_config,
        model_dir=args.model_dir,
        config_override=None,
        train_steps=args.train_steps,
        use_tpu=args.use_tpu,
        save_final_config=True,
        eval_train_data=args.eval_train_data,
        eval_config=eval_config
    )

if __name__ == '__main__':
    tf.compat.v1.app.run()
