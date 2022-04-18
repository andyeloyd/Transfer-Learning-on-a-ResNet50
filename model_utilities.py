import tensorflow as tf
from keras_vggface.vggface import VGGFace
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Input

from arcface_layer import ArcFace
import math

#https://stackoverflow.com/questions/50364706/massive-overfit-during-resnet50-transfer-learning
# https://keras.io/api/layers/normalization_layers/batch_normalization/

def get_transfer_model(nb_class=8631, arcface=False, m=0.5, train_batch_norm=False):
    """

    :param nb_class:            Numero de clases
    :param arcface:             True: la capa de clasificacion sera ArcFace, False: sera capa densa.
    :param m:                   Margen a agregar en ArcFace. Si no se usar ArcFace, este parametro es ignorado.
    :param train_batch_norm:   Congela las capas de normalizacion de lote.
                                En keras hay un comportamiento un tanto confuso de estas capas, pues
                                poseen pesos no entrenables que normalizan la entrada de acuerdo a la desviacion y media
                                de cada lote de alimentamiento al ser entrenadas, pero despues del entrenamiento
                                conservan tales pesos para normalizar imagenes de entrada en el modo de inferencia.
                                Durante transferencia de aprendizaje es necesario hacer que estas capas
                                aprendan, pues los datos no tendran las mismas caracteristicas.
                                (True para transferencia de aprendizaje, False para inferencia)
    :return:                    Modelo ResNet50 pre-entrenado en VggFace2 acoplado con capa final sin entrenar
                                para realizar transferencia de aprendizaje.
    """

    #vgg_model = VGGFace(model='resnet50', include_top=False)
    #last_layer = vgg_model.get_layer('avg_pool').output
    vgg_model = VGGFace(model='resnet50')
    last_layer = vgg_model.layers[-2].output

    if arcface:
        s = math.sqrt(2)*math.log(nb_class - 1)
        label_input = Input([], name='label_input')

        out = ArcFace(n_classes=nb_class, s=s, m=m)(last_layer, label_input)
        custom_vgg_model = Model((vgg_model.input, label_input), out)
    else:
        out = Dense(nb_class, activation='softmax', name='classifier')(last_layer)
        custom_vgg_model = Model(vgg_model.input, out)

    #new_model = tf.keras.models.Model(model.input, model.layers.output)
    #custom_vgg_model.trainable = False
    # Freezes original model
    for i in range(len(vgg_model.layers)-1):
        custom_vgg_model.layers[i].trainable=False

    if train_batch_norm:
        # Freeze the BatchNorm layers
        for layer in custom_vgg_model.layers:
            if "BatchNormalization" in layer.__class__.__name__:
                layer.trainable = True
                #layer.training= False
    return custom_vgg_model
