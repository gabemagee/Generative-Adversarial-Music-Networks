import random
import torchaudio
import os
import torch



def file_to_RNN_tensor(filename):
    tensor,sample_rate = torchaudio.load(filename)
    b = list(torch.split(tensor,sample_rate))[:-1]
    for i in b:
        print(i.size())
    b = torch.stack(b)
    return b


o = os.getcwd()
os.chdir(o+"/post/positive/")
a = file_to_RNN_tensor("0.wav")
os.chdir(o)


class discriminator_RNN(torch.nn.Module):

    def __init__(self,input_size,hidden_size,output_size):
        super(discriminator_RNN,self).__init__()
        self.hidden_size = hidden_size
        self.i2h = torch.nn.Linear(input_size+hidden_size,hidden_size)
        self.i2o = torch.nn.Linear(input_size+hidden_size,output_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.learning_rate = 0.005

    def forward(self, input, hidden):
        combined = torch.cat((input,hidden),1)
        hidden = self.i2h(combined)
        output = self.softmax(self.i2o(combined))
        return output,hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    def train_rnn(self,input_tensor,result_tensor):

        hidden = self.initHidden()

        self.zero_grad()

        for i in range(input_tensor.size()[0]):
            output, hidden = self.forward(input_tensor[i],hidden)

        loss = torch.nn.NLLLoss(output,result_tensor)
        loss.backward()
        for p in self.parameters():
            p.data.add_(-self.learning_rate,p.grad.data)

        return output, loss.item()



rnn = discriminator_RNN(10000,100,1)
print(rnn.parameters())



class Linear_Model(torch.nn.Module):



    def __init__(self,input_size):
        self = super.__init__()
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
things to think about

1. what ratio of samples should go in training versus validation?

2. what ratio of training set should be negative/positive

3. Model architecture?

 - tensor ->

4.

"""






def n_random_elts(lst,n):
    sample = []
    original = lst.copy()
    for k in range(n):
        i = random.randint(0,len(original)-1)
        selection = original[i]
        sample.append(selection)
        del original[i]
    return sample, original

a = directory_to_testing_set(.3,.7)

for i in a:
    print(i)
