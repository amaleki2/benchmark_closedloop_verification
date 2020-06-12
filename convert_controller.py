# This file produces onnx and nnet files from keras model files.

import keras2onnx
from keras.models import load_model

def keras_to_onnx(keras_model, out_file):
    model = load_model(keras_model)
    onnx_model = keras2onnx.convert_keras(model, model.name)
    keras2onnx.save_model(onnx_model, out_file)

def keras_to_nnet(keras_model, out_file):
    model = load_model(keras_model)
    w = model.get_weights()
    l = model.layers
    with open(out_file, "w") as f:
        n_layers = len(l)
        f.write("%d \n" % n_layers)
        for i in range(n_layers):
            f.write("%d, " % w[i * 2].shape[0])
        f.write("%d \n" % w[-1].shape[0])
        for i in range(5):
            f.write("0\n")
        for k in range(0, len(w), 2):
            n_rows, n_cols = w[k].shape
            for i in range(n_cols):
                for j in range(n_rows - 1):
                    f.write("%0.5f, " % w[k][j, i])
                f.write("%0.5f \n" % w[k][n_rows - 1, i])
            for i in range(n_cols):
                f.write("%0.5f \n" % w[k + 1][i])


keras_file = "controller_files/controller_triple_pendulum.h5"
onnx_file = "controller_files/controller_triple_pendulum.onnx"
nnet_file = "controller_files/controller_triple_pendulum.nnet"
keras_to_onnx(keras_file, onnx_file)
keras_to_nnet(keras_file, nnet_file)
