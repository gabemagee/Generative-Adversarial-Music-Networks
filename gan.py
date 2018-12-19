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
        b[i] = torch.reshape(b[i],tuple([10000])).type(torch.FloatTensor)
    b = torch.stack(b)
    return b

def list_of_tuples_to_tensors(lst):
    inputs = []
    outputs = []
    for item in lst:
        inputs.append(file_to_RNN_tensor(item[0]))
        outputs.append(torch.tensor([item[1]]).type(torch.FloatTensor))
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
        self.loss = torch.nn.L1Loss()

    def forward(self, input, hidden):
        combined = torch.cat((input,hidden),0)
        hidden = self.i2h(combined)
        #print(combined.size())
        presoftmax = self.i2o(combined)
        output = self.softmax(presoftmax)
        #print(output.size())
        return output,hidden


    def initHidden(self):
        return torch.zeros(self.hidden_size)

    def train_rnn(self,input_tensor,result_tensor):

        hidden = self.initHidden()

        self.zero_grad()

        for i in range(input_tensor.size()[0]):
            output, hidden = self.forward(input_tensor[i],hidden)
        #print(output.size(),result_tensor.size())
        loss = self.loss(output,result_tensor)
        loss.backward()
        for p in self.parameters():
            p.data.add_(-self.learning_rate,p.grad.data)

        return output, loss.item()

    def examine_forgery(self,input_tensor):
        hidden = self.initHidden()

        self.zero_grad()

        for i in range(input_tensor.size()[0]):
            output, hidden = self.forward(input_tensor[i], hidden)

        return output


class forger(torch.nn.Module):

    def __init__(self,input_size,hidden_size):
        super(forger,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = torch.nn.Linear(input_size+hidden_size,hidden_size)
        self.i2o = torch.nn.Linear(input_size+hidden_size,input_size*15)
        self.softmax = torch.nn.LogSoftmax(dim = 0)
        self.learning_rate = 0.005
        self.loss = torch.nn.L1Loss()

    def forward(self, input, hidden):
        combined = torch.cat((input,hidden),0)
        hidden = self.i2h(combined)
        #print(combined.size())
        presoftmax = self.i2o(combined)
        output = self.softmax(presoftmax)
        #print(output.size())
        return output,hidden

    def initHidden(self):
        return torch.zeros(self.hidden_size)

    def train_rnn(self,discriminator):

        hidden = self.initHidden()

        self.zero_grad()

        input_tensor = random_RNN_tensor(15)
        result_tensor = torch.Tensor([1]).type(torch.FloatTensor)
        for i in range(input_tensor.size()[0]):
            output, hidden = self.forward(input_tensor[i],hidden)
        b = list(torch.split(output, 15))[:-1]
        for i in range(len(b)):
            b[i] = torch.reshape(b[i], tuple([10000])).type(torch.FloatTensor)
        b = torch.stack(b)
        from_disc = discriminator(b)
        loss = self.loss(from_disc,result_tensor)
        loss.backward()
        for p in self.parameters():
            p.data.add_(-self.learning_rate,p.grad.data)

        return output, loss.item()

    def produce_clip(self,filename):
        hidden = self.initHidden()
        input_tensor = random_RNN_tensor(15)
        print(input_tensor)
        result_tensor = torch.Tensor([1]).type(torch.FloatTensor)
        for i in range(input_tensor.size()[0]):
            output, hidden = self.forward(input_tensor[i], hidden)
        print(output)
        output = torch.reshape(output,(150000,1))
        torchaudio.save(filename,output,10000)

def dir_to_tensors(directory):
    files = os.listdir(directory)
    os.chdir(directory)
    lst = []
    for file in files:
        t = file_to_RNN_tensor(file)
        lst.append(t)
    return lst

def audio_files_to_tensor_files(directory,destination):
    files = os.listdir(directory)
    for file in files:
        os.chdir(directory)
        t = file_to_RNN_tensor(file)
        os.chdir(destination)
        filename = file.split(".")[0]+".pt"
        torch.save(t, filename)



def random_RNN_tensor(size):
    b = []
    for i in range(size):
        b.append(torch.randint(low = -100000000,high = 100000000, size = tuple([10000])).type(torch.FloatTensor))
    b = torch.stack(b)
    return b

def files_to_samples():
    src = "/Volumes/Seagate Expansion Drive/NN/post"
    dest = "/Users/glma2016/Box/samples/tensors"
    os.chdir(src)
    print(src)
    j = 0
    for folder in os.listdir(src):
        j  += len(os.listdir(src + "/" + folder))
    print(j," total files")
    i = 0
    for folder in os.listdir(src):
        files = os.listdir(src+"/"+folder)
        for file in files:
            os.chdir(src + "/" + folder)
            b = file_to_RNN_tensor(file)
            filename = ".".join(file.split(".")[:-1])+".pt"
            #print(filename)
            os.chdir(dest+ "/" + folder)
            torch.save(b,filename)
            i +=1
            if i%1000==0:
                print(i/j)


def main():
    #divide samples in training and validation set
    hidden_size = 10000
    input_size = 10000
    iter = 1
    ratio_to_be_tested = .7
    positive_samples_directory = ""
    negative_samples_directory = ""
    disc = discriminator(input_size, hidden_size)
    frgr = forger(input_size, hidden_size)
    p_t, p_v = testing_and_validation(positive_samples_directory,ratio_to_be_tested)
    n_t, n_v = testing_and_validation(negative_samples_directory,ratio_to_be_tested)
    testing_set = zip_tuple(p_t,1) + zip_tuple(n_t,0)
    validation_set = zip_tuple(p_v,1) + zip_tuple(n_v,0)
    random.shuffle(testing_set)
    random.shuffle(validation_set)
    #training
    for i in range(iter):
        for sample,result in [testing_set[0]]:
            input = file_to_RNN_tensor(sample)
            result_tensor = torch.Tensor(tuple([result]))
            disc.train_rnn(input_tensor=input,result_tensor=result_tensor)









    #train discriminator
    #validation set test
    #GAN cycle
        #train forger for x iterations - save data
        #train discriminator for x iterations on data
        #produce wav files

if __name__ == '__main__':
    files_to_samples()
