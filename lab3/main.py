import time
import keras
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt

from eval import data_loader, eval

start = time.time()

cl_val_filename = './data/cl/valid.h5'
cl_test_filename = './data/cl/test.h5'

bd_val_filename = './data/bd/bd_valid.h5'
bd_test_filename = './data/bd/bd_test.h5'

model_filename = './models/bd_net.h5'
model_weights_filename = './models/bd_weights.h5'

b_prime_model_filename = './models/b_prime_model_%.2f.h5'

# load data
cl_val_x, cl_val_y = data_loader(cl_val_filename)
cl_test_x, cl_test_y = data_loader(cl_test_filename)
bd_val_x, bd_val_y = data_loader(bd_val_filename)
bd_test_x, bd_test_y = data_loader(bd_test_filename)
num_of_classes = len(np.unique(cl_test_y))

# BadNet B
bd_model = keras.models.load_model(model_filename)
# conv_3 -> pool_3 (last pooling layer)
# print(bd_model.summary())
cl_val_pred = np.argmax(bd_model.predict(cl_val_x), axis=1)
acc = accuracy_score(cl_val_y, cl_val_pred)

# New Network B' (B prime)
b_prime_model = keras.models.clone_model(bd_model)
b_prime_model.set_weights(bd_model.get_weights())

last_pooling_layer = b_prime_model.get_layer('conv_3')
last_pooling_model = keras.Model(inputs=b_prime_model.input, outputs=b_prime_model.get_layer('conv_3').output)
last_pooling_output = np.sum(last_pooling_model.predict(cl_val_x), axis=(0, 1, 2))
# channels should be removed in ascending order of average activation values
ascending_sorted_idx_list = np.argsort(last_pooling_output)

clean_accuracy_list = []
attack_success_rate_list = []
fraction_list = []
x_list = [.02, .04, .10, .30]
x_i = 0

for idx, del_idx in enumerate(ascending_sorted_idx_list):
    # no diff before the first 30 layers
    if idx < 30:
        last_pooling_layer_weights = np.array(last_pooling_layer.get_weights()[0])
        last_pooling_layer_weights[:, :, :, del_idx] = np.zeros(last_pooling_layer_weights.shape[:3])
        last_pooling_layer.set_weights(list([last_pooling_layer_weights, last_pooling_layer.get_weights()[1]]))
        continue

    del_start_time = time.time()

    last_pooling_layer_weights = np.array(last_pooling_layer.get_weights()[0])
    last_pooling_layer_weights[:, :, :, del_idx] = np.zeros(last_pooling_layer_weights.shape[:3])
    last_pooling_layer.set_weights(list([last_pooling_layer_weights, last_pooling_layer.get_weights()[1]]))

    clean_accuracy, attack_success_rate = eval(bd_model, b_prime_model, cl_test_x, cl_test_y, bd_test_x, bd_test_y, num_of_classes)
    fraction = (idx + 1) / len(ascending_sorted_idx_list)

    clean_accuracy_list.append(clean_accuracy)
    attack_success_rate_list.append(attack_success_rate)
    fraction_list.append(fraction)
    del_end_time = time.time()

    cl_val_pred = np.argmax(b_prime_model.predict(cl_val_x), axis=1)
    val_acc = accuracy_score(cl_val_y, cl_val_pred)
    if x_i < len(x_list) and acc - val_acc > x_list[x_i]:
        b_prime_model.save(b_prime_model_filename % x_list[x_i])
        # .30
        if x_i == 3:
            print('-' * 5, attack_success_rate)
        x_i += 1

    print('=' * 5, 'deleting layer: %d/%d, time cost: %.2fs' % (idx, len(ascending_sorted_idx_list), del_end_time-del_start_time))
    print('clean acc', clean_accuracy)
    print('attack success rate', attack_success_rate)

plt.plot(fraction_list, clean_accuracy_list, label='Clean Accuracy')
plt.plot(fraction_list, attack_success_rate_list, label='Attack Success Rate')
plt.xlabel('Fraction of Channels Pruned')
plt.ylabel('Accuracy')
plt.legend()
# plt.show()
plt.savefig('acc.png')

end = time.time()
print('time cost: %.2fs' % (end-start))
