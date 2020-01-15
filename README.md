# Generative Models Authorship

## Summary
The purpose of this project is to explore the extent upto which images generated using generative models
can be identied with their respective generator models. Two major techniques are used. One of them is to model this problem as a classication
task and use a Convolutional Neural Network to classify the images generated from respective GANs. The
other technique is to train an inverse model which tries to learn the latent space of the GAN. The image
is then reconstructed and compared with the original input image. If the L2 norm of the distance between
two images is less than a given threshold, then it is highly likely that the image is generated with the same Generator.

## Datasets
The datasets used are MNIST and CIFAR-10.

## Results
The results are documented in report.pdf