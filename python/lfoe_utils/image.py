import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import torch
import os

def extract_features(img_folder, id_to_img, img_to_id, config):
    device = torch.device('cuda' if config.get("use_cuda", True) and torch.cuda.is_available() else 'cpu')
    model_src = config["source"]
    model_name = config["model"]
    input_size = tuple(config["input_size"])
    mean = config["normalize_mean"]
    std = config["normalize_std"]

    vits16 = torch.hub.load(model_src, model_name)
    vits16.eval().to(device)

    img_names = [f for f in os.listdir(img_folder) if f in id_to_img.values()]
    max_img_id = max(img_to_id.values())

    with torch.no_grad():
        feat_size = vits16(torch.rand(1, 3, *input_size).to(device)).numel()
    img_to_feat = torch.zeros((max_img_id + 1, feat_size), dtype=torch.float32)

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    for img_name in img_names:
        img_path = os.path.join(img_folder, img_name)
        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = vits16(img)
        img_id = img_to_id[img_name]
        img_to_feat[img_id] = feats.view(-1).cpu()

    return img_to_feat

