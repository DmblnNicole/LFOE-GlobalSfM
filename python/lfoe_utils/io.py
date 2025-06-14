import numpy as np
import torch
import struct


def parse_viewgraph(file_path):
    id_pair_to_pose = {}
    id_to_img = {}
    img_to_id = {}
    with open(file_path, "rb") as f:
        while True:
            header = f.read(16)
            if len(header) < 16:
                break
            try:
                src_id, dst_id, src_len, dst_len = struct.unpack("iiII", header)
                src_name = f.read(src_len).decode("utf-8")
                dst_name = f.read(dst_len).decode("utf-8")
                rot_data = f.read(9 * 8)
                trans_data = f.read(3 * 8)
                if len(rot_data) < 72 or len(trans_data) < 24:
                    print("Bad pose entry")
                    break
                rot = np.frombuffer(rot_data, dtype=np.float64).reshape(3, 3).astype(np.float32)
                trans = np.frombuffer(trans_data, dtype=np.float64).astype(np.float32)
                rot = torch.tensor(rot).flatten()
                trans = torch.tensor(trans)
                id_pair_to_pose[(src_id, dst_id)] = [rot, trans, src_name, dst_name]
                if src_id not in id_to_img:
                    id_to_img[src_id] = src_name
                    img_to_id[src_name] = src_id
                if dst_id not in id_to_img:
                    id_to_img[dst_id] = dst_name
                    img_to_id[dst_name] = dst_id
            except Exception as e:
                print(f"Failed parsing entry: {e}")
                break
    return id_pair_to_pose, id_to_img, img_to_id

def parse_rotations(file_path):
    id_to_rot = {}
    with open(file_path, 'rb') as f:
        while True:
            id_bytes = f.read(4)
            if not id_bytes:
                break
            img_id = struct.unpack('i', id_bytes)[0]
            rot_bytes = f.read(72) 
            rot = np.frombuffer(rot_bytes, dtype=np.float64).reshape(3, 3)
            rot = torch.tensor(rot, dtype=torch.float32).flatten()
            id_to_rot[img_id] = rot
    max_id = max(id_to_rot.keys())
    id_to_rot_t = torch.zeros((max_id + 1, 9), dtype=torch.float32)
    for img_id, rot in id_to_rot.items():
        id_to_rot_t[img_id] = rot
    return id_to_rot_t


def parse_cutlabels(file_path):
    cut_labels = {}
    with open(file_path, 'r') as file:
        for line in file:
            k, v = map(int, line.strip().split())
            cut_labels[k] = v
    return cut_labels