# MatryoshkaNetworks
Experiments with deep convolutional generative models.

This repo contains code implementing the models described in the ICLR workshop submission "Learning to take data apart, and put it back together". These were implemented using Python and Theano. The main model is implemented in "MatryoshkaNetworks.py" and the sub-modules around which the model is built are all implemented in "MatryoshkaModules.py".

To run the tests you'll need to make a subdirectory "./mnist/data" inside the repo root directory, and put the \*.amat files for binarized MNIST into "./mnist/data/\*.amat". For training on dynamically-binarized data, you need to copy the standard UdM-style mnist.pkl.gz to "./mnist/data/mnist.pkl.gz".

You can find these files at:

http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat
http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat
http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat
http://deeplearning.net/data/mnist/mnist.pkl.gz


This repo started as a fork of the DCGAN code from: https://github.com/Newmu/dcgan_code. If you track back through the commit history, you can find some GAN experiments and hybrid GAN/VAE experiments that might be fun to play with. The Matryoshka Network is a bit tricky to get working as a GAN, seemingly due to the large number of latent variables. However, there are some tricks to get it training successfully. You can also train the model simultaneously as a VAE, with distribution matching on features from arbitrary layers in the discriminator.
