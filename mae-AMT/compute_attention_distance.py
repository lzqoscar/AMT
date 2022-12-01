import numpy as np
import functools
import matplotlib.pyplot as plt
import tqdm
# import pandas as pd


# @title Compute receptive field sizes

head_stride = 1
block_stride = 1
num_heads = attn_weights[0].shape[1]
img_w = 224
img_h = 224
patch_res = 14
factor = img_w / patch_res

# Show attention maps:
blocks = range(0, len(attn_weights), block_stride)
heads = range(0, num_heads, head_stride)

@functools.lru_cache()
def get_distance_map(query_col, query_row):
  x, y = np.meshgrid(
      np.linspace(0, img_w - 1.0 * factor, patch_res) - (query_col) * factor,
      np.linspace(0, img_h - 1.0 * factor, patch_res) - (query_row) * factor)
  return np.sqrt(x ** 2.0 + y ** 2.0)

# Compute distance maps:
maps = []
for query_row in range(0, patch_res):
  for query_col in range(0, patch_res):
    # Compute distance maps:
    maps.append(get_distance_map(query_col, query_row))
maps = np.stack(maps, axis=0)
maps = np.reshape(maps, [patch_res * patch_res, patch_res * patch_res])

attn_weights_np = [np.array(a) for a in attn_weights]
ymax = 0

df_data = []

for block in tqdm(blocks):
    for head in heads:

        all_extents = []
        for example in range(128):
            w = attn_weights_np[block][example, head, 1:, 1:]

            # Compute weighted mean of indivudal maps (weights sum to 1 because they are 
            # the output of a softmax):
            map_mean = np.sum(maps * w, axis=-1)

            # Compute mean across all query locations:
            query_mean = np.mean(map_mean)
            
            # Compute mean extent across all query locations:
            all_extents.append(query_mean)
  
        df_data.append(
            {
                'model_description': "VIT",
                'block': block,
                'head': head,
                'all_extents': np.array(all_extents)
            }
        )

# Plot figure
fig, ax = plt.subplots(1, 1)

df_plot = df_data.copy()
# model_description = df_plot.model_description[0]
model_description ="VIT"
df_plot = df_plot.pivot(index='block', columns='head', values='mean_extent')

labels = []
handles = []
for head in range(df_plot.shape[1]):
  h, = ax.plot(df_plot.iloc[:, head], '.')
  handles.append(h)
  labels.append(f'Head {head+1}' if head < 3 else '...')

ax.set_xlabel('Network depth (layer)')
if ax.is_first_col():
  ax.set_ylabel('Mean attention distance (pixels)')
ax.legend(handles[:4], labels[:4], loc=4)
# ax.set_title(pretty_names[model_description])
ax.set_title(model_description)

ax.set_ylim(0.0, 130)