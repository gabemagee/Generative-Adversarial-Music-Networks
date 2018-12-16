import random
import torchaudio
import os
import torch



class forger_RNN(torch.nn.Module):

        def __init__(self, tensor_size, hidden_size):
            super(forger_RNN, self).__init__()
            self.hidden_size = hidden_size
            self.i2h = torch.nn.Linear(tensor_size + hidden_size, hidden_size)
            self.i2o = torch.nn.Linear(tensor_size + hidden_size, tensor_size)
            self.softmax = torch.nn.LogSoftmax(dim=1)
            self.learning_rate = 0.005
            self.input_size = tensor_size

        def forward(self, input, hidden):
            combined = torch.cat((input, hidden), 1)
            hidden = self.i2h(combined)
            output = self.softmax(self.i2o(combined))
            return output, hidden

        def initHidden(self):
            return torch.zeros(1, self.hidden_size)

        def train_rnn(self, input_tensor, result_tensor):

            hidden = self.initHidden()

            self.zero_grad()

            for i in range(input_tensor.size()[0]):
                output, hidden = self.forward(input_tensor[i], hidden)


            loss = torch.nn.NLLLoss(output, result_tensor)
            loss.backward()
            for p in self.parameters():
                p.data.add_(-self.learning_rate, p.grad.data)

            return output, loss.item()

        def create_noise_input(self):
            return torch.rand(self.input_size,1)


rnn = forger_RNN(15001,1000,15001)
a = rnn.create_noise_input()
print(a)
print(a.size())