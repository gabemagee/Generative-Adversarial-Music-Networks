import torchaudio
import os
import torch

os.chdir(os.getcwd()+"/post/positive/")
audio_tensor, sample_rate = torchaudio.load('1.wav')
sz = audio_tensor.size()
print(sz,sample_rate)
print("-----")
audio_tensor, sample_rate = torchaudio.load('2.wav')
sz = audio_tensor.size()
print(sz,sample_rate)


"""
things to think about 

1. what ratio of samples should go in training versus validation?

2. what ratio of training set should be negative/positive

3. Model architecture?

 - tensor ->

4. 

"""

input_dimension = 15000000
output_dimension = 1
model=torch.nn.Sequential(torch.nn.Linear(input_dimension,output_dimension))

class RNN_Model(torch.nn.Module):

    def __init__(self):
        pass

class Linear_Model(torch.nn.Module):

    learning_rate = 0.0001

    def __init__(self,input_size):
        a = torch.nn.Linear()
        self.model = torch.nn.Linear()

