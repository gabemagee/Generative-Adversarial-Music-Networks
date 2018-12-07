import os
"""
import torchaudio
import torch
"""

"""
o = os.getcwd()
os.chdir(o+"/post/positive/")
audio_tensor, sample_rate = torchaudio.load('1.wav')
sz = list(audio_tensor.size())
print(sz[0],sample_rate)
print("-----")
audio_tensor, sample_rate = torchaudio.load('2.wav')
sz = audio_tensor.size()
print(sz,sample_rate)
os.chdir(o)

input_dimension = 15000000
output_dimension = 1
model=torch.nn.Sequential(torch.nn.Linear(input_dimension,output_dimension))

class RNN_Model(torch.nn.Module):

    def __init__(self):
        pass

class Linear_Model(torch.nn.Module):



    def __init__(self,input_size):
        second_layer_input = 100
        first_layer = torch.nn.Linear(input_size,second_layer_input)
        second_layer = torch.nn.Linear(second_layer_input,1)
        self.model = torch.nn.Sequential(first_layer, second_layer)
        self.learning_rate = 0.0001
        self.sr = 10000
        self.sz = [input_size,1]
        self.loss_fn = torch.nn.MSELoss(size_average=True)

    def evaluate(self,input_file):
        tensor, sample_rate = torchaudio.load(input_file)

        #check if the sample rate and size are correct
        assert(sample_rate != self.sr)
        assert(list(tensor.size())!= self.sz)

        return self.model(tensor)

    def train(self,iterations,training_set_directory):
        os.chdir(training_set_directory)
        subdirs = os.listdir()
        X = torch.Tensor()
        Y = torch.Tensor()

        for t in range(iterations):
            Yhatt = self.model(X)
            loss = self.loss_fn(Yhatt,Y)
            if t % 1000 == 0:
                print(loss.item())
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for param in model.parameters():
                    param -= self.learning_rate * param.grad

"""

"""
things to think about 

1. what ratio of samples should go in training versus validation?

2. what ratio of training set should be negative/positive

3. Model architecture?

 - tensor ->

4. 

"""




def directory_to_testing_set(training_set_directory):
    os.chdir(training_set_directory)
    subdirs = os.listdir()
    pos = os.getcwd()+"\\positive\\"
    neg = os.getcwd()+"\\negative\\"
    os.chdir(pos)
    postitive_samples = os.walk(pos)
    os.chdir(neg)
    negative_samples = os.walk(neg)
    #print(postitive_samples)
    #print(negative_samples)
    res = []
    for (dirpath, dirnames, filenames) in postitive_samples:
        if len(filenames)!=0:
            for file in filenames:
                res.append((dirpath+"\\"+file,1))
    for (dirpath, dirnames, filenames) in negative_samples:
        if len(filenames)!=0:
            for file in filenames:
                res.append((dirpath+"\\"+file,0))
    return res


l = directory_to_testing_set(os.getcwd()+"/pre/")

for i in l:
    print(i)