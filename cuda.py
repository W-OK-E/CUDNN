import torch
import math
import torch.functional as F

class LLTM(torch.nn.Module):
    def __init__(self,input_features,state_size):
        super(LLTM,self).__init__()
        self.input_features = input_features
        self.state_size = state_size


        self.weights = torch.nn.Parameter(
            torch.empty(3 * state_size, input_features + state_size))
        self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv =1.0/math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv,+stdv)
    
    def forward(self,input,state):
        old_h,old_cell = state
        X = torch.cat([old_h,input],dim = 1)

        gate_weights = F.linear(X,self.weights,self.bias)
        gates = gate_weights.chunk(3,dim = 1)
        input_gate = torch.sigmoid(gates[0])
        output_gate = torch.sigmoid(gates[1])

        candidate_cell = F.elu(gates[2])

        new_cell = old_cell + candidate_cell


lltm = LLTM(input_features=[12,34,45])
