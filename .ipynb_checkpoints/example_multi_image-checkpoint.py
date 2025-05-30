import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'auto'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import numpy as np
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils
import matplotlib.pyplot as plt

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

# Load an image
images = [
    Image.open("input/case5v1.png"),
    Image.open("input/case5v2.png"),
]

# Run the pipeline
outputs = pipeline.run_multi_image(
    images,
    seed=1,
    # Optional parameters
    sparse_structure_sampler_params={
        "steps": 24,
        "cfg_strength": 7.5,
    },
    slat_sampler_params={
        "steps": 24,
        "cfg_strength": 3,
    },
    mode='attention'
)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes

video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
imageio.mimsave("output/case5.mp4", video, fps=30)

attention_weights = pipeline.attention_weights
t = list(attention_weights.keys())
att = list(attention_weights.values())

att_front = [_[0].item() for _ in att]
att_back = [_[1].item() for _ in att]

plt.plot(t, att_front, label="front")
plt.plot(t, att_back, label="back")
# plt.set_xlim(1, 0)
plt.xlabel("timestep")
plt.ylabel("attention weight")
plt.legend()
plt.savefig("output/case5_att.png")
