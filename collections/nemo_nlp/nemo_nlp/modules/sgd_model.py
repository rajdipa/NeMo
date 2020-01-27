#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 20:56:22 2020

@author: ebakhturina
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(0)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 3)
        self.z = torch.tensor([[4.0,4.,4.]], requires_grad=True)

    def forward(self, x):
        x = torch.cat([self.z,x])
        x = F.relu(self.fc1(x))
        return x.view(-1, 3)
    
    
class Logits(nn.Module):
    def __init__(self,
                 num_classes,
                 embedding_dim):
        """Get logits for elements by conditioning on utterance embedding.

        Args:
          element_embeddings: A tensor of shape (batch_size, num_elements,
            embedding_dim).
          num_classes: An int containing the number of classes for which logits are
            to be generated.
          name_scope: The name scope to be used for layers.
    
        Returns:
          A tensor of shape (batch_size, num_elements, num_classes) containing the
          logits.
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.utterance_proj = nn.Linear(embedding_dim, embedding_dim)
        self.activation = torch.nn.functional.gelu
        
        self.layer1 = nn.Linear(2 * embedding_dim, embedding_dim)
        self.layer2 = nn.Linear(embedding_dim, num_classes)

    def forward(self,
                encoded_utterance,
                element_embeddings):
        
        """
        encoded_utterance - [CLS] token hidden state from BERT encoding of the utterance
        
        """
        _, num_elements, _ = [*element_embeddings.size()]
        
        # Project the utterance embeddings.
        utterance_embedding = self.utterance_proj(encoded_utterance)
        utterance_embedding = self.activation(utterance_embedding)
        
         # Combine the utterance and element embeddings.
        repeat_utterance_embedding = utterance_embedding.repeat(1,
                                                                num_elements,
                                                                1)
        utterance_element_emb = torch.cat([repeat_utterance_embedding,
                                           element_embeddings], axis = 2)
        logits = self.layer1(utterance_element_emb)
        logits = self.activation(logits)
        logits = self.layer2(logits)
        return logits


class SGDModel(nn.Module):
    def __init__(self,
                 num_classes,
                 embedding_dim):
        """Get logits for elements by conditioning on utterance embedding.

        Args:
          element_embeddings: A tensor of shape (batch_size, num_elements,
            embedding_dim).
          num_classes: An int containing the number of classes for which logits are
            to be generated.
          name_scope: The name scope to be used for layers.
    
        Returns:
          A tensor of shape (batch_size, num_elements, num_classes) containing the
          logits.
        """
        super().__init__()
        
        self.intent_layer = Logits(1, embedding_dim)

    def forward(self,
                encoded_utterance,
                intent_embeddings):
        
        """
        encoded_utterance - [CLS] token hidden state from BERT encoding of the utterance
        
        """
        intent_logits = self.intent_layer(encoded_utterance, intent_embeddings)
        
        # Shape: (batch_size, max_intents + 1)
        intent_logits = intent_logits.squeeze(axis=-1)
        return intent_logits
    
    

utter = torch.tensor([[1.,2.,3., 4., 5]])
num_classes = 1 # need to choose the most probable intent
num_elements = 4 # max num of intents per service
       
intent_embedding = torch.tensor([[1., 1., 1., 1., 1.],
                                 [2., 2., 2., 2., 2.],
                                 [0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0.]]).unsqueeze(0) # 1, 4, 5
    
embedding_dim = utter.size()[1]
net = SGDModel(num_classes=num_classes,
          embedding_dim=embedding_dim)
print(net)
logits = net(utter, intent_embedding)
print(logits)














