# This repository is an example of a GAN (Generative Adversarial Network). 
 GANs work by pitting two AI programs against eachother - one to spot true and fake examples of a certain type of media or object. It is the discriminator. The other creates fake objects to try to trick the discriminator. It is the forger. Eventually, these two can theoretically reach an equilibrium. The majority of the program is located in gan.py
 
### Discriminator
Recursively iterates over a tensor generated from a 15-second .wav file - looking for similar characteristics that it can find that signify a musical clip, returning a single digit classifier. 1 for yes, 0 for no. It also has methods to evaluate but not train, mostly for the purpose of analyzing the forger's work  

### Forger
Starts with random inputs and generates what could be a music file, passes it to the discriminator. Eventually over enough iterations, it optimizes to maximize the output of the discriminator. In other words, it learns how to make files that the discriminator considers music.

### Other
Also in this file are other dependencies such as audiotorch-using methods I made to convert .wav files into RNN-acceptable tensors. Another method I have is one that divides the testing and validation sets up randomly by a preconceived ratio so we can evaluate our model after we train it.

### Examples of products

Pending!
