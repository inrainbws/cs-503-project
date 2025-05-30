# From Photos to Avatars: Stylized 3D Modeling from 2D Images

Reconstructing stylized 3D models from 2D images is a challenging problem with significant applications in digital content creation, gaming, and avatar generation. However, this problem often suffers from inconsistent views across multi-view 2D inputs, leading to undesirable artifacts in the resulting 3D reconstructions. To address this issue, we propose a two-stage approach: first, generating stylized multi-view 2D images using GPT-4o, and second, constructing 3D models from these images. To overcome cross-view inconsistency and the Janus effect, we propose an attention-guided diffusion model sampling strategy, which enforces coherence across views and ensures more accurate 3D geometry. This work highlights both the potential and the challenges of stylized 3D modeling pipelines and paves the way for producing high-quality, fabrication-ready 3D outputs from minimal 2D input.

### Installation

    Create a new conda environment named `trellis` and install the dependencies:
    ```sh
    . ./setup.sh --new-env --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
    ```
    The detailed usage of `setup.sh` can be found by running `. ./setup.sh --help`.
    ```sh
    Usage: setup.sh [OPTIONS]
    Options:
        -h, --help              Display this help message
        --new-env               Create a new conda environment
        --basic                 Install basic dependencies
        --train                 Install training dependencies
        --xformers              Install xformers
        --flash-attn            Install flash-attn
        --diffoctreerast        Install diffoctreerast
        --vox2seq               Install vox2seq
        --spconv                Install spconv
        --mipgaussian           Install mip-splatting
        --kaolin                Install kaolin
        --nvdiffrast            Install nvdiffrast
        --demo                  Install all dependencies for demo
    ```


<!-- Usage -->
## Usage

Generate 3D model conditioned on 2D stylized input
```
python example_multi_image.py
```

Saving 3D model
```
# GLB files can be extracted from the outputs
glb = postprocessing_utils.to_glb(
    outputs['gaussian'][0],
    outputs['mesh'][0],
    # Optional parameters
    simplify=0.95,          # Ratio of triangles to remove in the simplification process
    texture_size=1024,      # Size of the texture used for the GLB
)
glb.export("sample.glb")
```



## Mesh Processing with ManifoldPlus and MeshLab

We store the mesh representations from Trellis to `.obj` files.
To convert a raw `.obj` mesh into a clean, watertight manifold suitable for 3D printing or simulation, we use [ManifoldPlus](https://github.com/hjwdzh/ManifoldPlus). First, install ManifoldPlus following the instructions in the repository. Then, you can process an `.obj` file using the provided Python or command-line interface. For example:
```./ManifoldPlus input.obj output.obj```
This command reads the input mesh, performs automatic watertight remeshing, and outputs a closed, oriented manifold mesh. Further manual adjustment is done with [MeshLab](https://www.meshlab.net/).

