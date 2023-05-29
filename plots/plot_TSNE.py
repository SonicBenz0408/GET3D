import pathlib

import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold
from sklearn.decomposition import PCA

if __name__ == '__main__':
    tex_dir = pathlib.Path('/home/mason/Mason/VLLAB/GET3D/save_inference_results/car-chair/inference/hook/mapping')
    geo_dir = pathlib.Path('/home/mason/Mason/VLLAB/GET3D/save_inference_results/car-chair/inference/hook/mapping_geo')

    z_tex = np.load(tex_dir / 'input.npy')
    c_tex = np.load(tex_dir / 'c.npy')
    w_tex = np.load(tex_dir / 'output.npy')[:,0,:]

    z_geo = np.load(geo_dir / 'input.npy')
    c_geo = np.load(geo_dir / 'c.npy')
    w_geo = np.load(geo_dir / 'output.npy')[:,0,:]

    # print(z.shape, c.shape, w.shape)

    X = np.concatenate([w_tex])
    Y = np.concatenate([c_tex])

    X_tsne = manifold.TSNE(n_components=2, init='random', verbose=1).fit_transform(X)

    # Data Visualization
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize
    plt.figure(figsize=(16, 16))
    
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(Y[i]), color=plt.cm.prism(Y[i]*10), 
                fontdict={'weight': 'bold', 'size': 9})
    
    plt.savefig('tmp.png')
    