from uwnet import *
def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

def norm_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
            make_batchnorm_layer(8),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
            make_batchnorm_layer(16),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
            make_batchnorm_layer(32),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)


print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

# print("making model...")
# batch = 128
# iters = 500
# rate = .01
# momentum = .9
# decay = .005

# m = conv_net()
# print("training...")
# train_image_classifier(m, train, batch, iters, rate, momentum, decay)
# print("done")
# print

# print("evaluating model...")
# print("training accuracy: %f", accuracy_net(m, train))
# print("test accuracy:     %f", accuracy_net(m, test))


print("making model...")
batch = 128
momentum = .9
decay = .005

m2 = norm_net()
print("training...")
for rate in [.1, .05, .025, .01, .005]:
    print(rate)
    train_image_classifier(m2, train, batch, 100, .01, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m2, train))
print("test accuracy:     %f", accuracy_net(m2, test))

# 7.6 Question: What do you notice about training the convnet with/without batch normalization?
# How does it affect convergence?
# How does it affect what magnitude of learning rate you can use?
# Write down any observations from your experiments:
# 
# convnet train accuracy: .405
# convnet test accuracy: .403
# normnet train accuracy: .534
# normnet test accuracy: .522
#
# we noticed that it takes a little longer to train with normalization, but noticed 
# that we were able to get fairly stable weight updates (small changes in loss per iteration) 
# even with an initial learning rate of .1.
# 
# we found that even without modifying learning rate, we were able to get far better test and train
# accuracy than without normalization in the 500 epochs we trained each model for.
#
# without normalization, using a learning rate of .1 led to very noisy updates, with
# significant increases and decreases in learning loss between iterations. With normalization
# these updates were much less noisy, but loss was still decreasing quickly on average, indicating
# that the larger learning rate makes the early stages of training more efficient by allowing
# larger steps towards minima without as much random fluctuation.
