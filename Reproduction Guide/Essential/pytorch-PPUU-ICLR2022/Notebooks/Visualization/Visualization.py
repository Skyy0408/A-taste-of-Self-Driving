# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import DimensionalityReduction
import Widgets
import DataReader
import VisualizationLibrary
import importlib
import Tabs
from types import ModuleType

importlib.reload(DataReader)
importlib.reload(DimensionalityReduction)
importlib.reload(Widgets)
importlib.reload(Tabs)
importlib.reload(VisualizationLibrary)

v = VisualizationLibrary.Visualization()
v.display()

# %%
import sklearn
import sklearn.cluster
from matplotlib import pyplot as plt

features = PCA.PCA.get_pca_data()
scores = []

for c in range(2, 15):
    score = sklearn.cluster.KMeans(n_clusters=c).fit(features).score(features)
    scores.append(-score)

plt.plot(scores, 'o-')




# %% [markdown]
# # Just one episode 

# %%
import DimensionalityReduction
import Widgets
import DataReader
import VisualizationLibrary
import importlib
import Tabs
from types import ModuleType
import torch

importlib.reload(DataReader)
importlib.reload(DimensionalityReduction)
importlib.reload(Widgets)
importlib.reload(Tabs)
importlib.reload(VisualizationLibrary)


# %%

for i in range(0, 385):
    x = torch.load(f'/home/us441/nvidia-collab/vlad/results/mpc/interactive_eval/best_mpc/train/episode_data/{i}')
    if not x['result']['road_completed']:
        print(i)


# %%
x = torch.load(f'/home/us441/nvidia-collab/vlad/results/mpc/interactive_eval/best_mpc/train/episode_data/1')

# %%
x['result'].keys()


# %%
def adapt(xs):
    res = {
        'state_sequences' : [],
        'action_sequences' : [],
        'images' : [],
    }
            
    mapping = {
        'state_sequences' : 'state_sequence',
        'action_sequences' : 'action_sequence',
        'images' : 'images',
    }
    
    for x in xs:
        for k, v in mapping.items():
            res[k].append(x[v])
    return res

v = Widgets.EpisodeVisualizer(adapt([x]))
v

# %%
import cv2
import numpy as np
import imageio as iio

def images_to_video(images, path):
    size = (117, 24)
    fps = 10
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'x264'), fps, (size[1], size[0]), False)
    for i in range(images.shape[0]):
        img = np.random.randint(0, 255, (*size, 3), dtype = np.uint8)
#         out.write(images[i].permute(1, 2, 0).numpy())
        out.write(img)
    out.release()
    
def images_to_video_2(images, path):
    w = iio.get_writer(path, format='FFMPEG', mode='I', fps=10,
                       codec='h264_vaapi',
#                        output_params=['-vaapi_device',
#                                       '/dev/dri/renderD128',
#                                       '-vf',
#                                       'format=gray|nv12,hwupload'],
                       pixelformat='vaapi_vld')
    for i in range(images.shape[0]):
        w.append_data(images[i].permute(1, 2, 0).numpy())
        print(i)
        
    w.append_data(images[i].permute(1, 2, 0).numpy())
        
    w.close()
    
def images_to_video_3(images, path):
    w = iio.get_writer(path, mode='I')
    for i in range(images.shape[0]):
        w.append_data(images[i].permute(1, 2, 0).numpy())
        print(i)
        
    w.close()
    
images_to_video(x['images'], '/home/us441/check.mp4')
iio.mimwrite('/home/us441/check.gif', x['images'].permute(0, 2, 3, 1).numpy(), fps=20)

# %%
#########################################
# using cell magic
#########################################
from IPython.display import HTML

HTML("""
<img src='/home/us441/check.gif'/>
""")


# %%
x['result']

# %%

HTML("""
<div align="middle">
<video width="80%" controls>
      <source src="/home/us441/check.mp4" type="video/mp4">
</video></div>
""")

# %%
