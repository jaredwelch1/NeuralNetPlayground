# Neural Net Playground

Collection of notebooks created while learning about NerualNets

## Current NN architectures

- Recurrent Neural Network

    RNNs are good for pattern recognition. They use something call LSTM, which stands for Long Short Term Memory. Basically, RNNs use 
    pattern data to train at predicting the next value in a sequence. The RNN in this repo is designed to predict the next character
    given a pattern of characters before it.

- CNN/Resnet

    There is jupyter notebook in the tutorial stuff for a convolutional neural net that gets 99% on the MNIST handwritten 
    digits dataset. CNNs are great at taking features from image data and tranforming them into deep networks that can be trained
    on image classification tasks. 
    
    Resnet takes these CNN architectures a bit further, using residual learning to improve training and results. Links to some resnet 
    papers can be found in the readme in the resnet directory if futher reading is desired on the topic of resnets. 

- Generative Adversarial Network 

    GANs are a relatively new type of deep learning architecture that uses two different networks, one that generates output data from
    some input vector noise, and another network that takes input data and decides whether it is a real example or a generated one. The 
    training interaction between the two networks creates a powerful generator for new images that can look almost identical to real ones.
    
    Some applications of GANs are image generation, faceswapping images and videos, image style transfer, and many more. 
    
## Resources

- RNN tutorial from deep learning without PHD
    - [Link to slides](https://docs.google.com/presentation/d/e/2PACX-1vRouwj_3cYsmLrNNI3Uq5gv5-hYp_QFdeoan2GlxKgIZRSejozruAbVV0IMXBoPsINB7Jw92vJo2EAM/pub#slide=id.p)
    - [Link to code on github](https://github.com/martin-gorner/tensorflow-rnn-shakespeare)

- GAN tutorial github project
    - [link](https://github.com/uclaacmai/Generative-Adversarial-Network-Tutorial)
