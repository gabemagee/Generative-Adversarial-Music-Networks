import random
import torchaudio
import os
import torch
#from pathlib import Path

def n_random_elts(lst,n):
    sample = []
    original = lst[:]
    for k in range(n):
        i = random.randint(0,len(original)-1)
        selection = original[i]
        sample.append(selection)
        del original[i]
    return sample, original

def testing_and_validation(directory,ratio):
    files = list(os.listdir(directory))
    for i in range(len(files)):
        files[i] = directory+"/"+files[i]
    n = len(files)
    size = int(n*ratio)
    testing_set, validation_set = n_random_elts(files,size)
    return testing_set,validation_set

def zip_tuple(lst,value):
    ret = []
    for item in lst:
        ret.append((item,value))
    return ret

def file_to_RNN_tensor(filename):
    tensor,sample_rate = torchaudio.load(filename)
    b = list(torch.split(tensor,sample_rate))[:-1]
    for i in range(len(b)):
        b[i] = torch.reshape(b[i],tuple([10000]))
    b = torch.stack(b)
    return b

def list_of_tuples_to_tensors(lst):
    inputs = []
    outputs = []
    for item in lst:
        inputs.append(file_to_RNN_tensor(item[0]))
        outputs.append(torch.tensor([item[1]]))
    return inputs,outputs


class discriminator(torch.nn.Module):

    def __init__(self,input_size,hidden_size):
        super(discriminator,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = torch.nn.Linear(input_size+hidden_size,hidden_size)
        self.i2o = torch.nn.Linear(input_size+hidden_size,1)
        self.softmax = torch.nn.LogSoftmax(dim = 0)
        self.learning_rate = 0.005
        self.loss = torch.nn.NLLLoss()

    def forward(self, input, hidden):
        combined = torch.cat((input,hidden),0)
        hidden = self.i2h(combined)
        #print(combined.size())
        presoftmax = self.i2o(combined)
        output = self.softmax(presoftmax)
        print(output.size())
        return output,hidden

    def initHidden(self):
        return torch.zeros(self.hidden_size)

    def train_rnn(self,input_tensor,result_tensor):

        hidden = self.initHidden()

        self.zero_grad()

        for i in range(input_tensor.size()[0]):
            output, hidden = self.forward(input_tensor[i],hidden)
        print(output.size(),result_tensor.size())
        loss = self.loss(output,result_tensor)
        loss.backward()
        for p in self.parameters():
            p.data.add_(-self.learning_rate,p.grad.data)

        return output, loss.item()


def main():
    #divide samples in training and validation set
    hidden_size = 10000
    directory = os.getcwd()
    ratio = 0.7
    assert('post' in list(os.listdir(directory)))
    os.chdir(directory+"/post")
    folders = os.listdir(os.getcwd())
    assert('positive' in folders)
    assert('negative' in folders)
    p = os.getcwd()+'/positive'
    n = os.getcwd()+'/negative'
    t_p, v_p = testing_and_validation(p,ratio)
    t_n, v_n = testing_and_validation(n,ratio)
    testing_set = zip_tuple(t_p,1) + zip_tuple(t_n,0)
    validation_set = zip_tuple(v_p,1) + zip_tuple(v_n,0)
    random.shuffle(testing_set)
    random.shuffle(validation_set)
    testing_tensors = list_of_tuples_to_tensors(testing_set)
    validation_tensors = list_of_tuples_to_tensors(validation_set)
    #print(testing_tensors[0][0])
    input_size = list(testing_tensors[0][0].size())[1]
    disc = discriminator(input_size,hidden_size)
    i = 0
    for i in range(len(testing_tensors[0])):
        output,loss = disc.train_rnn(testing_tensors[0][i],testing_tensors[1][i])
        print(output,loss)



    #train discriminator
    #validation set test
    #GAN cycle
        #train forger for x iterations - save data
        #train discriminator for x iterations on data
        #produce wav files

if __name__ == '__main__':
    main()
