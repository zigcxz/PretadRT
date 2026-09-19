import torch
import torch.nn as nn


from torch_geometric.nn import (
    GINConv,
    global_mean_pool
)



class GINEncoder(nn.Module):


    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        output_dim=256
    ):

        super().__init__()


        nn1=nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )


        self.conv1=GINConv(
            nn1
        )



        nn2=nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )


        self.conv2=GINConv(
            nn2
        )


        self.fc=nn.Linear(
            hidden_dim,
            output_dim
        )



    def forward(
        self,
        x,
        edge_index,
        batch
    ):


        x=self.conv1(
            x,
            edge_index
        )


        x=torch.relu(x)


        x=self.conv2(
            x,
            edge_index
        )


        x=torch.relu(x)



        x=global_mean_pool(
            x,
            batch
        )


        return self.fc(x)