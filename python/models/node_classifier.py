import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class NodeClassifier(torch.nn.Module): 
    # 0 (inlier), 1 (outlier)
    def __init__(self, config):
        super().__init__()
        self.conv1 = TransformerConv(
            in_channels=config["in_channels"], 
            out_channels=6, 
            edge_dim=config["edge_dim"], 
            heads=config["heads"], 
            beta=config["beta_conv1"], 
            dropout=config["dropout_conv1"]
        )
        self.conv2 = TransformerConv(
            in_channels=config["heads"] * 6, 
            out_channels=3, 
            edge_dim=config["edge_dim"], 
            heads=config["heads"], 
            beta=config["beta_conv2"], 
            dropout=config["dropout_conv2"]
        )
        self.conv3 = TransformerConv(
            in_channels=config["heads"] * 3, 
            out_channels=config["hidden_dim"],
            edge_dim=config["edge_dim"], 
            heads=1, 
            beta=config["beta_conv3"], 
            dropout=config["dropout_conv3"]
        )
        self.classifier = torch.nn.Linear(config["hidden_dim"], 1)
        self.dropout_1 = config["dropout_1"]
        self.dropout_2 = config["dropout_2"]

    def forward(self, x, edge_index, edge_attr):
        h = self.conv1(x, edge_index, edge_attr)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout_1, training=self.training)

        h = self.conv2(h, edge_index, edge_attr)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout_2, training=self.training)

        h = self.conv3(h, edge_index, edge_attr)
        h = F.relu(h)

        logits = self.classifier(h).squeeze(-1)  # shape: [num_nodes]
        return logits