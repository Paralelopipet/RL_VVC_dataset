import torch
from torch.distributions.categorical import Categorical


n = torch.argmax(torch.tensor([ 0.25, 0.25, 0.75, 0.25 ])) # return a tensor with the highest probability
m = Categorical(torch.tensor([ 0.25, 0.25, 0.75, 0.25 ])) # return a  random tensor from  a categorical distribution to help the agent get random action.
a= n  # equal probability of 0, 1, 2, 3
b= m.sample()  # equal probability of 0, 1, 2, 3

print(a)
print(b)