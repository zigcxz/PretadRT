import torch.nn as nn



class ProjectionHead(nn.Module):


    def __init__(
        self,
        dim=256,
        projection_dim=128
    ):

        super().__init__()


        self.net=nn.Sequential(

            nn.Linear(
                dim,
                dim
            ),

            nn.ReLU(),

            nn.Linear(
                dim,
                projection_dim
            )
        )


    def forward(self,x):

        return self.net(x)