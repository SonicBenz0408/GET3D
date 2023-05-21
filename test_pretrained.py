import torch

model = torch.load("/home/mason/VLLAB/GET3D/log/00008-stylegan2-motorbike-chair-car-gpus1-batch32-gamma80/network-snapshot-002662.pt")

for weight in model['G']:
    if weight in ['mapping.fc0.weight', 'mapping.fc0.bias', 'mapping_geo.fc0.weight', 'mapping_geo.fc0.bias']:
        model['G'][weight] = torch.concat([model['G'][weight], model['G'][weight]])
    print(weight, model['G'][weight].shape)