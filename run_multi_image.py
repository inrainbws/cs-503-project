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

# Load a pipeline from a model folder or a Hugging Face model hub.
# pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline = TrellisImageTo3DPipeline.from_pretrained("gqk/TRELLIS-image-large-fork")
pipeline.cuda()

# Load an image
images = [
    Image.open("input/case3v1.png"),
    Image.open("input/case3v2.png"),
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

mesh = outputs['mesh'][0]  # Get the MeshExtractResult
vertices = mesh.vertices.cpu().numpy()  # shape: [N, 3]
faces = mesh.faces.cpu().numpy()        # shape: [M, 3]

def save_obj(filename, vertices, faces):
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            # OBJ is 1-indexed
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

save_obj("output/output_mesh.obj", vertices, faces)
print("Mesh saved to output/output_mesh.obj")

EXTENSION_HOME = "extensions"
manifoldplus = f"{EXTENSION_HOME}/ManifoldPlus/build/manifold"

os.system(f"{manifoldplus} --input output/output_mesh.obj --output output/output_mesh_manifold.obj")
print("Manifold mesh saved to output/output_mesh_manifold.obj")