import random
import torchaudio
import os

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


things to think about 

1. what ratio of samples should go in training versus validation?

2. what ratio of training set should be negative/positive

3. Model architecture?

 - tensor ->

4. 

input_dimension = 15000000
output_dimension = 1
model=torch.nn.Sequential(torch.nn.Linear(input_dimension,output_dimension))

class RNN_Model(torch.nn.Module):

    def __init__(self,input_size,hidden_size,output_size):
        super(RNN_Model,self).__init__()
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

rnn = RNN_Model(100000,100,1)
print(rnn.parameters())

def train_rnn(rnn,input_tensor,result_tensor):

    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(input_tensor.size()[0]):
        output, hidden = rnn(input_tensor[i],hidden)

    loss = torch.nn.NLLLoss(output,result_tensor)
    loss.backward()
    for p in rnn.parameters():
        p.data.add_(-rnn.learning_rate,p.grad.data)

    return output, loss.item()




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

def directory_to_testing_set(positive_ratio,testing_ratio):
    all_samples = os.getcwd() + "/post/"
    negative = all_samples + "/negative/"
    positive = all_samples + "/positive/"
    negative_sample_lst = os.listdir(negative)
    positive_sample_lst = os.listdir(positive)
    negative_sample_testing = int(len(negative_sample_lst)*testing_ratio)
    positive_sample_testing = int(len(positive_sample_lst)*testing_ratio)
    positive_samples = n_random_elts(positive_sample_lst,positive_sample_testing)
    negative_samples = n_random_elts(negative_sample_lst,negative_sample_testing)

    validation_pos = positive_samples[1]
    validation_neg = negative_samples[1]
    testing_pos = positive_samples[0]
    testing_neg = negative_samples[0]

    a = int((len(validation_neg)+len(validation_pos))*positive_ratio)
    b = len(validation_neg)+len(validation_pos) - a
    if a <= len(validation_pos) and b < len(validation_neg):
        pv = n_random_elts(validation_pos,a)[0]
        nv = n_random_elts(validation_neg,b)[0]
    elif a > len(validation_pos) and b < len(validation_neg):
        pv = validation_pos
        n = (len(validation_pos)/positive_ratio)*(1-positive_ratio)
        nv = n_random_elts(validation_neg,n)[0]
    elif b > len(validation_neg) and a <= len(validation_pos):
        nv = validation_neg
        n = (len(validation_neg) / (1-positive_ratio)) * positive_ratio
        pv = n_random_elts(validation_pos,n)[0]
    else:
        raise Exception("division error validation set")


    a = int((len(testing_neg)+len(testing_pos))*positive_ratio)
    b = len(testing_neg)+len(testing_pos) - a
    if a <= len(testing_pos) and b < len(testing_neg):
        pt = n_random_elts(testing_pos,a)[0]
        nt = n_random_elts(testing_neg,b)[0]
    elif a > len(testing_pos) and b < len(testing_neg):
        pt = testing_pos
        n = (len(testing_pos)/positive_ratio)*(1-positive_ratio)
        nt = n_random_elts(testing_neg,n)[0]
    elif b > len(testing_neg) and a <= len(testing_pos):
        nt = testing_neg
        n = (len(testing_neg) / (1-positive_ratio)) * positive_ratio
        pt = n_random_elts(testing_pos,n)[0]
    else:
        raise Exception("division error validation set")


    for i in range(len(pt)):
        pt[i] = (positive + pt[i],1)
    for i in range(len(nt)):
        nt[i] = (negative + nt[i],0)
    for i in range(len(pv)):
        pv[i] = (positive + pv[i],1)
    for i in range(len(nv)):
        nv[i] = (negative + nv[i],0)
    l = [pt,nt,pv]
    return l




def n_random_elts(lst,n):
    sample = []
    original = lst.copy()
    for k in range(n):
        i = random.randint(0,len(original)-1)
        selection = original[i]
        sample.append(selection)
        del original[i]
    return sample, original

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
