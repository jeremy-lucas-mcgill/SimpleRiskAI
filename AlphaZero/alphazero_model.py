import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader

class AlphaZeroModel(nn.Module):
    def __init__(self,input_size,hidden_size,action_size):
        super().__init__()

        #parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.action_size = action_size

        #define first layer
        self.fc1 = nn.Linear(input_size,hidden_size)

        #define policy head
        self.fc_policy = nn.Linear(hidden_size,action_size)

        #define value head
        self.fc_value = nn.Linear(hidden_size,1)
    
    def forward(self,x):
        output = F.relu(self.fc1(x))
        p = self.fc_policy(output)
        v = self.fc_value(output)
        return v, p
    
    def sample_action(self,state):
        with torch.no_grad():
            v, p = self.forward(state)
            return v, F.softmax(p,dim=-1)
    
    def action_logits(self,state):
        with torch.no_grad():
            v, p_logits = self.forward(state)
            return v, p_logits
    
    def loss(self,v,p, z, pi, model_params,c=1e-4):
        #calculate value loss
        value_loss = (v - z) ** 2

        #calculate policy loss
        policy_loss = -torch.sum(pi * torch.log(p), dim=-1)

        #calculate regularization loss
        L2_regularization = c * sum((param**2).sum() for param in model_params)

        #sum all losses and return average
        total_loss = value_loss + policy_loss + L2_regularization

        return total_loss.mean()
    
    def train_model(self,s,z,pi,learning_rate=0.001,batch_size=64,epochs=10):
        #set optimizer
        optimizer = torch.optim.Adam(self.parameters(),lr=learning_rate)

        #create dataset and dataloader
        dataset = TensorDataset(s,z,pi)
        data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

        #start training
        for epoch in range(epochs):
            #reset running loss
            running_loss = 0
            
            #iterate through batches from dataloader
            for index, (s,z,pi) in enumerate(data_loader):
                #clear previous gradients
                optimizer.zero_grad()
                
                #get predictions from network
                v,p = self.forward(s)

                #apply softmax
                p = F.softmax(p, dim=-1)

                #calculate loss and update network
                loss = self.loss(v,p,z,pi,self.parameters())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(data_loader)}")


