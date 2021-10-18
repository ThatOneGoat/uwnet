from uwnet import *

mnist = False

inputs = 784 if mnist else 3072

def softmax_model():
    l = [make_connected_layer(inputs, 10),
        make_activation_layer(SOFTMAX)]
    return make_net(l)

def neural_net():
    l = [   make_connected_layer(inputs, 512),
            make_activation_layer(RELU),
            make_connected_layer(512, 256),
            make_activation_layer(RELU),
            make_connected_layer(256, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
if mnist:
    train = load_image_classification_data("mnist/mnist.train", "mnist/mnist.labels")
    test  = load_image_classification_data("mnist/mnist.test", "mnist/mnist.labels")
else:
    train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
    test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .001

m = neural_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# We reached about .975 accuracy on MNIST data.

# Our accuracy on CIFAR is much lower than on mnist. The best test accuracy
# we could reach is around .51.

# We found that the same parameters for rate, momentum, iters, and decay 
# worked well for both problem spaces. However, we found that making just one
# layer with many neurons improved accuracy a lot for MNIST, whereas adding
# more layers improved accuracy more on CIFAR. We think this is because the CIFAR
# problem space is more difficult, so the model benefits more from additional
# complexity from the extra activation layers.