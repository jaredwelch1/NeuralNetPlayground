# Resnet

Reset ( short for deep residual neural network ) is an improvement on previous DNN architectures, allowing deeper networks
and faster training. 

In depth reading about resnets can be found here: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

This directory will contain a python tensorflow implementation of a resnet, with scripts for:
- input data preparation for feeding into the model
- model definition and structure 
- training of the model on a dataset
- evaluation of model accuracy on a validation dataset 

To accomplish this, I plan to use a [custom estimator function](https://www.tensorflow.org/get_started/custom_estimators) to become
more familiar with tensorflow estimator implementation 
