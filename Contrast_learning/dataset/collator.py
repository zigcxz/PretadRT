from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch



def contrastive_collate(batch):


    anchors=[]
    augs=[]


    indices=[]


    for item in batch:

        anchors.append(
            item["anchor"]
        )

        augs.append(
            item["aug"]
        )

        indices.append(
            item["idx"]
        )



    return {

        "anchor":
        Batch.from_data_list(
            anchors
        ),


        "aug":
        Batch.from_data_list(
            augs
        ),


        "indices":
        indices

    }