import tf2onnx
import tensorflow as tf
import os
import onnx

# https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#export-from-tf

tf_saved_model = 'C:\\Users\\Andrés\\Documents\\Resnet50_new_imp\\saved_model'
onnx_save_path = 'C:\\Users\\Andrés\\Documents\\Resnet50_new_imp\\onnx_model\\temp.onnx'


# Conversion de tf2 a onnx
# (Commando de linea llamado por medio de os.system)
cmd_prompt = 'python -m tf2onnx.convert --saved-model ' + tf_saved_model + ' --output ' + onnx_save_path
print(cmd_prompt)
os.system(cmd_prompt)


# Definicion de tamaño de lote explicito para optimizar tiempo de ejecución
onnx_model = onnx.load_model(onnx_save_path)
BATCH_SIZE = 1
inputs = onnx_model.graph.input
for input in inputs:
    dim1 = input.type.tensor_type.shape.dim[0]
    dim1.dim_value = BATCH_SIZE

model_name = 'C:\\Users\\Andrés\\Documents\\Resnet50_new_imp\\onnx_model\\resnet50_onnx_model.onnx'
onnx.save_model(onnx_model, model_name)
print('Onnx with explicit batch size of 1 written to %s' %model_name)
