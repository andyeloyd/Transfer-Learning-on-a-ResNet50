import tensorflow as tf
from keras_vggface.vggface import VGGFace
import cv2 as cv
import numpy as np
import os
import pandas as pd
from model_utilities import get_transfer_model
from datasets import get_dataset_from_directory
from tfrec_datasets import dataset_from_shards

from tensorflow.keras.layers import Input, BatchNormalization
from tensorflow.keras.layers import Flatten, Dense, Dropout


############ Definicion de parametros de entrenamiento y directorios de almacenamiento de archivos.##################
n_classes = 500
batch_size = 64
epochs = 35
m = 0.5
arcface = False
# Carpeta maestra para almacenamiento de archivos
#master_save_path = 'C:\\Users\\Andrés\\Documents\\Resnet50_new_imp_dense_500_ids'
master_save_path = 'C:\\Users\\andye\\Desktop\\M\\Tesis\\S5\\Finished projects\\new_try'
if not os.path.isdir(master_save_path):
    os.makedirs(master_save_path)
weights_save_path = os.path.join(master_save_path, 'weights_transfer_arcface_classes_%d_m_%d.h5' % (n_classes, m))
#weights_save_path = os.path.join(master_save_path, 'weights_transfer_%d.h5' % (n_classes))

################### Definicion de localizacion de conjuntos de datos ####################################
# Entrenamiento con 24 ids sin TFrecords
# dataset_path = 'C:\\cleaned_personal_dataset'

# Entrenamiento con 500 ids, usando TFRecords
tfrecords_dir = 'C:\\Users\\Andrés\\Documents\\vggface2_test_shards\\shards'
train_tfrec_name = 'vggface2_test_trainDS_224x224'
val_tfrec_name = 'vggface2_test_trainDS_224x224'

################# Obtencion de conjuntos de datos para entrenamiento y validacion##########################
# train_dataset, val_dataset = get_dataset_from_directory(dataset_path=dataset_path, batch_size=batch_size,
#                                                        bright_augment=True, split=0.33)
# Datos en formato de imagenes
# train_dataset, val_dataset = get_dataset_from_directory(dataset_path=dataset_path, batch_size=batch_size,
#                                                        bright_augment=True, split=0.33, arcface_format=arcface)
# Datos en formato TFRecord
train_dataset = dataset_from_shards(tfrecords_dir, train_tfrec_name, batch_size=batch_size)
val_dataset = dataset_from_shards(tfrecords_dir, val_tfrec_name, batch_size=batch_size)


#####################  Definicion del modelo de la red ###################################################
# model = get_transfer_model(nb_class=n_classes, train_batch_norm=True)
model = get_transfer_model(nb_class=n_classes, arcface=arcface, m=m, train_batch_norm=True)
model.summary()

# Definicion de optimizador para entrenamiento y compilado de la red
# optimizer ='adam'
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=False)
loss = 'sparse_categorical_crossentropy'
model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')


######################## Definicion de callbacks ############################################################
#       Tensorboard para registrar cambios de parametros de la red
#       EarlyStopping para detener el entrenamiento una vez la red deje de mejorar su perdida
#       Checkpoints para guardar el modelo en el estado con mejores resultados
tensorboard_logdir = os.path.join(master_save_path, 'tensorboard')
callback_tensorboard = tf.keras.callbacks.TensorBoard(tensorboard_logdir, histogram_freq=1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3)
chkpts_filename = os.path.join(master_save_path, 'checkpoints', 'weights-{epoch:03d}-{val_loss:.4f}.hdf5')
callback_checkpoints = tf.keras.callbacks.ModelCheckpoint(chkpts_filename, monitor='val_loss',
                                                          verbose=1, save_best_only=True)
callbacks = [callback_tensorboard, early_stopping, callback_checkpoints]

######################## Entrenamiento del modelo ###########################################################
if arcface:
    print('Training model with ArcFace layer.')
else:
    print('Training model with Dense layer.')
model.fit(x=train_dataset, epochs=epochs, callbacks=callbacks, validation_data=val_dataset)

# Guardado de pesos del modelo entrenado a disco
model.save(weights_save_path)
print('Saved model weights to %s' %weights_save_path)

