from uwnet import *

def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(186, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)
def fc_net():
    l = [   make_connected_layer(3072, 18),
            make_activation_layer(RELU),
            make_connected_layer(18, 18),
            make_activation_layer(RELU),
            make_connected_layer(18, 18),
            make_activation_layer(RELU),
            make_connected_layer(18, 18),
            make_activation_layer(RELU),
            make_connected_layer(18, 18),
            make_activation_layer(RELU),
            make_connected_layer(18, 18),
            make_activation_layer(RELU),
            make_connected_layer(18, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .005

m = conv_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))
# To calculate the number of matrix operations, we looked at the convolutional layers
# and the fully connected layer at the end. For convolution layers, we assumed one data
# point as input and used the following formula (see maxe_convolutional_layer param names):
# ops = filters * (c * size ^ 2) *((w - 1) / stride + 1) * (( h- 1) / stride + 1) ).
# This formula corresponds to the dimensions of the matrices being multiplied. For the 
# FC layer, we used simply use ops = inputs * outputs. Applying these formulas and adding
# for each layer gave us a total of 1108480 operations.
# 
# 
# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
#

