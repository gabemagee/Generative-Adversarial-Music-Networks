import random
#import torchaudio
import time
import os
import torch
import numpy
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

def remove_non_tensors(lst):
    lst = lst.copy()
    i = 0
    while i < len(lst):
        if lst[i].split(".")[-1]!='pt':
            del lst[i]
        else:
            i+=1
    return lst

def testing_and_validation(directory,ratio):
    files = list(os.listdir(directory))
    files = remove_non_tensors(files)
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
"""
def file_to_RNN_tensor(filename):
    tensor,sample_rate = torchaudio.load(filename)
    b = list(torch.split(tensor,sample_rate))[:-1]
    for i in range(len(b)):
        b[i] = torch.reshape(b[i],tuple([10000])).type(torch.FloatTensor)
    b = torch.stack(b)
    return b
"""


def list_of_tuples_to_tensors(lst):
    inputs = []
    outputs = []
    for item in lst:
        inputs.append(file_to_RNN_tensor(item[0]))
        outputs.append(torch.tensor([item[1]]).type(torch.FloatTensor))
    return inputs,outputs


class discriminator(torch.nn.Module):

    def __init__(self,input_size,hidden_size,device='cpu'):
        super(discriminator,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = 0.005
        self.softmax = torch.nn.LogSoftmax(dim=0).to(device)
        self.i2h = torch.nn.Linear(input_size+hidden_size,hidden_size).to(device)
        self.i2o = torch.nn.Linear(input_size+hidden_size,1).to(device)
        self.loss = torch.nn.L1Loss().to(device)
        self.device = device

    def forward(self, input, hidden):
        combined = torch.cat((input,hidden),0).to(self.device)
        hidden = self.i2h(combined).to(self.device)
        #print(combined.size())
        presoftmax = self.i2o(combined).to(self.device)
        output = self.softmax(presoftmax).to(self.device)
        #print(output.size())
        return output,hidden


    def initHidden(self):
        return torch.zeros(self.hidden_size).to(self.device)

    def train_rnn(self,input_tensor,result_tensor):

        hidden = self.initHidden()

        self.zero_grad()

        for i in range(input_tensor.size()[0]):
            output, hidden = self.forward(input_tensor[i],hidden)
        loss = self.loss(output,result_tensor).to(self.device)
        loss.backward()
        for p in self.parameters():
            p.data.add_(-self.learning_rate,p.grad.data)

        return output, loss.item()

    def examine_forgery(self,input_tensor):
        hidden = self.initHidden()
        for i in range(input_tensor.size()[0]):
            output, hidden = self.forward(input_tensor[i], hidden)

        return output


class forger(torch.nn.Module):

    def __init__(self, input_size, hidden_size,device='cpu'):
        super(forger, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = torch.nn.Sequential(torch.nn.Linear(input_size + hidden_size,100).to(self.device),torch.nn.ReLU().to(device), torch.nn.Linear(100,hidden_size).to(self.device)).to(self.device)
        self.i2o = torch.nn.Linear(input_size + hidden_size, input_size).to(self.device)
        self.softmax = torch.nn.LogSoftmax(dim=0).to(self.device)
        self.learning_rate = 0.005
        self.loss = torch.nn.L1Loss().to(self.device)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 0)
        hidden = self.i2h(combined).to(self.device)
        # print(combined.size())
        presoftmax = self.i2o(combined).to(self.device)
        output = self.softmax(presoftmax).to(self.device)
        # print(output.size())
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.hidden_size).to(self.device)

    def train_rnn(self, discriminator):

        hidden = self.initHidden()

        self.zero_grad()

        input_tensor = random_RNN_tensor(15).to(self.device)
        result_tensor = torch.Tensor([1]).type(torch.FloatTensor).to(self.device)
        b = []
        for j in range(15):
            for i in range(input_tensor.size()[0]):
                output, hidden = self.forward(input_tensor[i], hidden)
            output = torch.reshape(output, tuple([10000])).type(torch.FloatTensor).to(self.device)
            b.append(output)
            input_tensor = b.copy()
            for i in range(14-j):
                input_tensor.append(torch.randint(low=-100000000, high=100000000, size=tuple([10000])).type(torch.FloatTensor).to(self.device))
            input_tensor = torch.stack(input_tensor)
        b = torch.stack(b).to(self.device)
        output = discriminator.examine_forgery(b)
        loss = self.loss(output, result_tensor).to(self.device)
        loss.backward()
        for p in self.parameters():
            p.data.add_(-self.learning_rate, p.grad.data)
        return b, loss.item()

    def produce_tensor(self, filename):
        hidden = self.initHidden()
        input_tensor = random_RNN_tensor(15).to(self.device)
        for i in range(input_tensor.size()[0]):
            output, hidden = self.forward(input_tensor[i], hidden)
        output = torch.reshape(output, (150000, 1)).to(self.device)
        torch.save(output, filename)
        # torchaudio.save(filename,output,10000)

def random_RNN_tensor(size):
    b=[]
    for i in range(size):
        b.append(torch.randint(low = -100000000,high = 100000000, size = tuple([10000])).type(torch.FloatTensor))
    b = torch.stack(b)
    return b

"""
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
"""


def divide_into_n(lst,n):
    m = len(lst)
    p = int(m/n)
    r = []
    for i in range(0,m,p):
        r.append(lst[i:i+p])
    return r

def load_model(filename,disc_frgr):
    device = torch.device("cuda")
    if disc_frgr=="discriminator":
        a = discriminator
    elif disc_frgr=="forger":
        a = forger
    else:
        raise Exception("choose discriminator or forger")
    model = a(*args, **kwargs)
    model.load_state_dict(torch.load(filename))
    model.to(device)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #dtype = torch.cuda.float if torch.cuda.is_available() else torch.float
    #device = torch.device('cpu')
    # divide samples in training and validation set
    hidden_size = 1000
    input_size = 10000
    epochs = 4
    count = 0
    frgr_training_iter = 1000
    ratio_to_be_tested = .7
    saved_model_directory = os.getcwd() + "\\models"
    positive_samples_directory = r"C:\Users\GMagee1\Box\positive"
    negative_samples_directory = r"C:\Users\GMagee1\Box\negative"
    created_wav_tensors_directory = r"C:\Users\GMagee1\Box\wavs"
    t = time.time()
    disc = discriminator(input_size, hidden_size,device).to(device)
    frgr = forger(input_size, hidden_size,device).to(device)
    print(time.time() - t)
    p_t, p_v = testing_and_validation(positive_samples_directory, ratio_to_be_tested)
    n_t, n_v = testing_and_validation(negative_samples_directory, ratio_to_be_tested)
    testing_set = zip_tuple(p_t, 1) + zip_tuple(n_t, 0)
    validation_set = zip_tuple(p_v, 1) + zip_tuple(n_v, 0)

    random.shuffle(validation_set)
    # training
    s = []
    total = (epochs*len(testing_set)//2)
    for i in range(epochs):
        t = time.time()
        j = 0
        random.shuffle(testing_set)
        for sample, result in testing_set[:len(testing_set)//2]:
            #print(sample)
            t0 = time.time()
            input = torch.load(sample,map_location=device)
            result_tensor = torch.Tensor(tuple([result])).cuda()
            _,l = disc.train_rnn(input_tensor=input, result_tensor=result_tensor)
            s.append(l)
            #print("load: ",t-t0," train: ",time.time()-t)
            j+=1
            if j%500==0:
                percentage = int((j/total)*100)
                print(percentage,"% done training.",j," examples seen.")
                print("Average Loss ",sum(s)/len(s))
                s = []
        print(time.time() - t, len(testing_set))
        torch.save(disc, saved_model_directory+"\\discriminators\\"+str(i)+".pt")


    # validation
    avgs = []
    for sample, result in validation_set:
        input = torch.load(sample).to(device)
        o = disc.examine_forgery(input).data.cpu().numpy()[0]
        #print(o)
        avgs.append(numpy.abs(result - o))
    print("Average loss: ",sum(avgs)/len(avgs))

    for j in range(epochs):
        created = []
        for i in range(frgr_training_iter):
            a, _ = frgr.train_rnn(disc)
            created.append((a, 0))
        torch.save(frgr, saved_model_directory + "\\forgers\\" + str(j) + ".pt")
        for t in created:
            torch.save(t,created_wav_tensors_directory+"\\"+str(count)+".pt")
            count+=1
        random.shuffle(created)
        for sample, result in created:
            # print(sample)
            input = sample.to(device)
            result_tensor = torch.Tensor(tuple([result])).to(device)
            disc.train_rnn(input_tensor=input, result_tensor=result_tensor)
        torch.save(disc, saved_model_directory + "\\discriminators\\" + str(i+epochs) + ".pt")
        #save dicriminator here











    #train discriminator
    #validation set test
    #GAN cycle
        #train forger for x iterations - save data
        #train discriminator for x iterations on data
        #produce wav files

if __name__ == '__main__':
    main()
