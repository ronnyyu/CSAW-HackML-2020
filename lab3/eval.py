import keras
import sys
import h5py
import numpy as np
import cv2


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data


def eval(bd_model, b_prime_model, cl_test_x, cl_test_y, bd_test_x, bd_test_y, num_of_classes):
    # clean accuracy: cl_test
    bd_pred = np.argmax(bd_model.predict(cl_test_x), axis=1)
    b_prime_pred = np.argmax(b_prime_model.predict(cl_test_x), axis=1)
    g_output_list = []
    for (bd_label, gd_label) in zip(bd_pred, b_prime_pred):
        g_output = bd_label if bd_label == gd_label else num_of_classes + 1
        g_output_list.append(g_output)
    clean_accuracy = np.mean(np.equal(g_output_list, cl_test_y)) * 100

    # attack accuracy: bd_test
    bd_pred = np.argmax(bd_model.predict(bd_test_x), axis=1)
    b_prime_pred = np.argmax(b_prime_model.predict(bd_test_x), axis=1)
    g_output_list = []
    for (bd_label, gd_label) in zip(bd_pred, b_prime_pred):
        g_output = bd_label if bd_label == gd_label else num_of_classes + 1
        g_output_list.append(g_output)
    attack_success_rate = np.mean(np.equal(g_output_list, bd_test_y)) * 100

    return clean_accuracy, attack_success_rate


def main():
    # bd_model_file = str(sys.argv[1])
    # b_prime_model_file = str(sys.argv[2])
    # image_file = str(sys.argv[3])
    #
    # bd_model = keras.models.load_model(bd_model_file)
    # b_prime_model = keras.models.load_model(b_prime_model_file)
    # image = cv2.imread(image_file)
    # image = image[np.newaxis, :]

    # by default, use the pretrained models and the first image of the dataset
    bd_model_file = './models/bd_net.h5'
    b_prime_model_file = './models/b_prime_model_0.02.h5'
    bd_test_filename = './data/bd/bd_test.h5'
    bd_test_x, bd_test_y = data_loader(bd_test_filename)
    num_of_classes = len(np.unique(bd_test_y))
    image = bd_test_x[0]
    image = image[np.newaxis, :]

    bd_model = keras.models.load_model(bd_model_file)
    b_prime_model = keras.models.load_model(b_prime_model_file)

    bd_pred = np.argmax(bd_model.predict(image), axis=1)
    b_prime_pred = np.argmax(b_prime_model.predict(image), axis=1)
    pred = 0
    if bd_pred == b_prime_pred:
        pred = bd_pred
    else:
        pred = num_of_classes + 1
    print(pred)
    return pred

if __name__ == '__main__':
    main()