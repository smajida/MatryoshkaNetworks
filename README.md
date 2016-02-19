# MatryoshkaNetworks
Experiments with deep convolutional generative models.

This repo contains code implementing the models described in the ICLR workshop submission "Learning to take data apart, and put it back together". The code is all Python and Theano. The main model is implemented in "MatryoshkaNetworks.py" and the various modules around which the model is built are all implemented in "MatryoshkaModules.py".

To run the tests you'll need to make a subdirectory "./mnist/data" inside the repo root directory and put the \*.amat files for binarized MNIST into "./mnist/data/\*.amat". For training on dynamically-binarized data, you'll need to copy the standard UdM-style mnist.pkl.gz to "./mnist/data/mnist.pkl.gz".

You can find these files at:

http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat
http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat
http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat
http://deeplearning.net/data/mnist/mnist.pkl.gz

Tests for the convolutional MatNet are implemented in "TestMNIST_conv.py". The main test options are towards the top of the file. Everything's reasonably well commented, but you'll have to read some code to do anything useful with these files. Sorry. For the convolutional model, you're best off with a K40 or TitanX due to high memory overhead. I'd recommend the TitanX, as the K40 is quite slow. You can probably get this going on a GTX 980ti or something like that if you reduce the batch size from 200 to 100 or whatever.

The convolutional model surpasses the previous state of the art after 1-2 hours (20-30 epochs), but you'll have to wait a day or two for the scores to get crazy. You can evaluate the trained model using "EvalMNIST_conv.py", which you'll have to edit to match the various options you set when training the model.

Evaluating the convolutional model on the test set is rather slow, due to the use of 2500 samples in the IWAE bound. This step is also pretty memory intensive, as we increase the batch size to 500 to reduce stochasticity induced by batch normalization. The evaluation script takes ~4 hours to complete on a TitanX.

Stuff for the fully-connected model is similarly arranged, in the files "TestMNIST_fc.py" and "EvalMNIST_fc.py". These tests are much faster than those for the convolutional model.

The tests for comparing fully-connected MatNets with various numbers of modules in their constructor networks are implemented in "TestMNIST_sanity_check.py" and "EvalMNIST_sanity_check.py".

This repo started as a fork of the DCGAN code from: https://github.com/Newmu/dcgan_code. If you track back through the commit history, you can find some GAN experiments and hybrid GAN/VAE experiments that might be fun to play with. The Matryoshka Network is a bit tricky to get working as a GAN, seemingly due to the large number of latent variables. However, there are some tricks to get it training successfully. You can also train the model simultaneously as a VAE, with distribution matching on features from arbitrary layers in the discriminator.
