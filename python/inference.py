from lfoe_utils.io import parse_cutlabels, parse_viewgraph, parse_rotations
from lfoe_utils.image import extract_features
from lfoe_utils.graph import linegraph, tuple_to_index, node_attr, edge_attr, create_subgraphs
from models.node_classifier import NodeClassifier
from torch_geometric.data import Data
import numpy as np
import yaml
import torch
import gc


def majority_voting(votes):
    outlier_edges = []
    for node, p in votes.items():
        if sum(p) > len(p)/2:
            outlier_edges.append(node)
    return outlier_edges


def init_voting_array(id_pair_to_pose):
    votes = {}
    for e in id_pair_to_pose.keys():
        votes[e] = []
    return votes


def preds_to_view_ids(preds, lg):
    v = np.asarray(lg.nodes)
    v = np.array([sorted(e_pair) for e_pair in v])
    return v[preds]


def run(data, lg, path_nc, config, sigmoid_thresh):
    device = torch.device('cuda' if config.get("use_cuda", True) and torch.cuda.is_available() else 'cpu')
    nc = NodeClassifier(config=config).to(device)
    nc.load_state_dict(torch.load(path_nc, map_location=device))
    nc.eval()
    data = data.to(device)
    with torch.no_grad():
        out = nc(data.x, data.edge_index, data.edge_attr)
        out = torch.sigmoid(out).view(-1)
    preds = out >= sigmoid_thresh
    outlier_edges = preds_to_view_ids(preds.cpu(), lg)
    return preds.cpu(), outlier_edges


def main_inference(path_images, path_clusters, path_viewgraph, path_rots, path_nc): 
    with open("../config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # get glomap viewgraph
    id_pair_to_pose, id_to_img, img_to_id = parse_viewgraph(path_viewgraph)
    cam_to_rot = parse_rotations(path_rots)
    
    # extract image embeddings
    img_embeds = extract_features(path_images, id_to_img, img_to_id, config["img_embeds"])
    
    # init voting array to resolve double predictions for edges
    votes = init_voting_array(id_pair_to_pose)

    # predict outliers for each subgraph
    cam_to_label = parse_cutlabels(path_clusters)
    subgraphs = create_subgraphs(cam_to_label, id_pair_to_pose)
    
    for i, subgraph in enumerate(subgraphs):
        # get linegraph
        v = subgraph[0]
        e = subgraph[1]
        lg = linegraph(v, e)
        v_attr = node_attr(lg, id_pair_to_pose)
        e_attr = edge_attr(lg, cam_to_rot, img_embeds)

        # normalize edge attr
        n = img_embeds.shape[1]
        e_attr_i = e_attr[:, :n]
        e_attr_r = e_attr[:, n:]
        ea_mean = e_attr_i.mean(dim=0, keepdim=True)
        ea_std = e_attr_i.std(dim=0, keepdim=True) + 1e-6 
        e_attr_i = (e_attr_i - ea_mean) / ea_std
        e_attr = torch.cat([e_attr_i, e_attr_r], dim=1)

        # create data
        e_idx_lg = tuple_to_index(lg)
        data = Data(x=v_attr, edge_index=e_idx_lg, edge_attr=e_attr)

        # run inference
        preds, outlier_edges_sub = run(data, lg, path_nc, config["node_classifier"], sigmoid_thresh=0.5)
        
        nodes= list(lg.nodes)
        for i in range(len(nodes)):
            node = tuple(sorted(nodes[i]))
            votes[node].append(int(preds[i]))
        
        # clean memory
        del data
        gc.collect()

    # majority voting on the final votes array
    outlier_edges = majority_voting(votes)
    return outlier_edges