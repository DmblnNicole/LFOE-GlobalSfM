import networkx as nx
import numpy as np
import torch


def linegraph(cameras, edge_idx):
    G = nx.Graph()
    G.add_nodes_from(cameras)
    G.add_edges_from(edge_idx)
    linegraph = nx.line_graph(G)
    return linegraph


def tuple_to_index(lg):
    v = torch.tensor(list(lg.nodes()), dtype=torch.long)
    e = torch.tensor(list(lg.edges()), dtype=torch.long)
    max_val = v.max().item() + 1
    vf = v[:, 0] * max_val + v[:, 1]
    idx = -torch.ones(vf.max().item() + 1, dtype=torch.long)
    idx[vf] = torch.arange(v.size(0))
    e = e[:, :, 0] * max_val + e[:, :, 1]
    e_idxs = idx[e].T
    return e_idxs
    

def node_attr(lg, id_pair_to_pose):
    v_attr = [
        torch.cat([
            id_pair_to_pose[tuple(sorted(n))][0],
            id_pair_to_pose[tuple(sorted(n))][1]
        ]) for n in lg.nodes
    ]
    return torch.stack(v_attr, dim=0)


def edge_attr(lg, cam_to_rot, cam_to_feat):
    e = np.array(lg.edges)
    v1 = e[:, 0]
    v2 = e[:, 1]
    v1e = v1[:, :, np.newaxis]
    v2e = v2[:, np.newaxis, :]
    matches = v1e == v2e
    matches = np.any(matches, axis=2)
    common_v = np.where(matches[:, 0], v1[:, 0], v1[:, 1])
    rots = cam_to_rot[common_v]
    feats = cam_to_feat[common_v]
    return torch.cat([rots, feats], dim=1)

    
def create_subgraphs(cam_to_label, id_pair_to_pose):
    subgraphs = []
    labels = set(cam_to_label.values())
    for label in labels:
        cams = []
        edges = set()
        for cam in cam_to_label.keys():
            if cam_to_label[cam] == label:
                # cameras contains all cameras belonging to current label
                cams.append(cam)
                for id_pair, _ in id_pair_to_pose.items():
                    if cam == id_pair[0] or cam == id_pair[1]:
                        # edges contains all edges connected to current label
                        edges.add(id_pair)
        subgraphs.append((cams, edges))
    return subgraphs