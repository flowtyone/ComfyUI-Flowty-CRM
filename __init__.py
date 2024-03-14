import sys
from os import path

sys.path.insert(0, path.dirname(__file__))

from folder_paths import get_filename_list, get_full_path, get_save_image_path, get_output_directory
from comfy.model_management import get_torch_device
from comfy.utils import ProgressBar
from PIL import Image
import numpy as np
import torch
import json
from .crmlib.model import CRM
from .crmlib.inference import generate3d, generate3d_cuda
from omegaconf import OmegaConf

class CRMPoserConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "processed_image":("IMAGE",),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 0xffffffffffffffff}),
                "cfg": ("FLOAT", {"default": 5.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "steps": ("INT", {"default": 30, "min": 6, "max": 10000}),
            }
        }

    RETURN_TYPES = ("CRM_POSE_CONFIG",)
    FUNCTION = "configure"
    CATEGORY = "Flowty CRM"

    def configure(self, processed_image, seed, cfg, steps):
        return ({"ref_image": processed_image, "seed": seed, "cfg": cfg, "steps": steps},)


class CRMPoseSampler:
    def __init__(self):
        self.initialized_poser = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixel_model": (get_filename_list("checkpoints"),),
                "config": ("CRM_POSE_CONFIG",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "Flowty CRM"

    def sample(self, pixel_model, config):
        device = get_torch_device()

        if not torch.cuda.is_available():
            device = "cpu"

        if not self.initialized_poser:
            print("Loading pixel model")

            from .crmlib.imagedream.ldm.util import (
                instantiate_from_config,
                get_obj_from_str,
            )

            stage1_config = path.join(path.dirname(__file__), "crmlib/configs", "nf7_v3_SNR_rd_size_stroke.yaml")

            stage1_config = OmegaConf.load(stage1_config).config
            stage1_sampler_config = stage1_config.sampler
            stage1_model_config = stage1_config.models
            stage1_model_config.resume = get_full_path("checkpoints", pixel_model)
            stage1_model_config.config = path.join(path.dirname(__file__), "crmlib/imagedream/configs",
                                                   "sd_v2_base_ipmv_zero_SNR.yaml")

            stage1_model = instantiate_from_config(OmegaConf.load(stage1_model_config.config).model)
            stage1_model.load_state_dict(torch.load(stage1_model_config.resume, map_location="cpu"), strict=False)
            stage1_model = stage1_model.to(device).to(torch.float32)
            stage1_model.device = device
            self.initialized_poser = get_obj_from_str(stage1_sampler_config.target)(
                stage1_model, device=device, dtype=torch.float32, **stage1_sampler_config.params
            )
            self.initialized_poser.seed = config.get("seed")

        ref_image = Image.fromarray(np.clip(255. * config.get("ref_image")[0].detach().cpu().numpy(), 0, 255).astype(np.uint8), "RGB")

        pbar = ProgressBar(int(config.get("steps")))
        p = {"prev": 0}

        def prog(i):
            i = i + 1
            if i < p["prev"]:
                p["prev"] = 0
            pbar.update(i - p["prev"])
            p["prev"] = i

        uc = self.initialized_poser.model.get_learned_conditioning([
            "uniform low no texture ugly, boring, bad anatomy, blurry, pixelated,  obscure, unnatural colors, poor lighting, dull, and unclear."
        ]).to(device)

        stage1_images = self.initialized_poser.i2i(
            self.initialized_poser.model,
            self.initialized_poser.size,
            "3D assets",
            uc=uc,
            sampler=self.initialized_poser.sampler,
            ip=ref_image,
            step=config.get("steps"),
            scale=config.get("cfg"),
            batch_size=self.initialized_poser.batch_size,
            ddim_eta=0.0,
            dtype=self.initialized_poser.dtype,
            device=self.initialized_poser.device,
            camera=self.initialized_poser.camera,
            num_frames=self.initialized_poser.num_frames,
            pixel_control=(self.initialized_poser.mode == "pixel"),
            transform=self.initialized_poser.image_transform,
            offset_noise=self.initialized_poser.offset_noise,
            callback=prog
        )

        stage1_images.pop(self.initialized_poser.ref_position)

        return (torch.stack([
            torch.from_numpy(img / 255.0) for img in stage1_images
        ]),)

class CCMSampler:
    def __init__(self):
        self.initialized_ccm = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ccm_model": (get_filename_list("checkpoints"),),
                "config": ("CRM_POSE_CONFIG",),
                "poses": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    CATEGORY = "Flowty CRM"

    def sample(self, ccm_model, config, poses):
        device = get_torch_device()

        if not torch.cuda.is_available():
            device = "cpu"

        if not self.initialized_ccm:
            print("Loading ccm model")

            from .crmlib.imagedream.ldm.util import (
                instantiate_from_config,
                get_obj_from_str,
            )

            stage2_config = path.join(path.dirname(__file__), "crmlib/configs", "stage2-v2-snr.yaml")

            stage2_config = OmegaConf.load(stage2_config).config
            stage2_sampler_config = stage2_config.sampler

            stage2_model_config = stage2_config.models

            stage2_model_config.resume = get_full_path("checkpoints", ccm_model)
            stage2_model_config.config = path.join(path.dirname(__file__), "crmlib/imagedream/configs",
                                                   "sd_v2_base_ipmv_chin8_zero_snr.yaml")

            stage2_model = instantiate_from_config(OmegaConf.load(stage2_model_config.config).model)
            sd = torch.load(stage2_model_config.resume, map_location="cpu")
            stage2_model.load_state_dict(sd, strict=False)
            stage2_model = stage2_model.to(device).to(torch.float32)
            stage2_model.device = device
            self.initialized_ccm = get_obj_from_str(stage2_sampler_config.target)(
                stage2_model, device=device, dtype=torch.float32, **stage2_sampler_config.params
            )
            self.initialized_ccm.seed = config.get("seed")

        ref_image = Image.fromarray(np.clip(255. * config.get("ref_image")[0].detach().cpu().numpy(), 0, 255).astype(np.uint8), "RGB")

        poses = [
            Image.fromarray(np.clip(255. * pose.detach().cpu().numpy(), 0, 255).astype(np.uint8)) for pose in poses
        ]

        pbar = ProgressBar(int(config.get("steps")))
        p = {"prev": 0}

        def prog(i):
            i = i + 1
            if i < p["prev"]:
                p["prev"] = 0
            pbar.update(i - p["prev"])
            p["prev"] = i

        stage2_images = self.initialized_ccm.i2iStage2(
            self.initialized_ccm.model,
            self.initialized_ccm.size,
            "3D assets",
            self.initialized_ccm.uc,
            self.initialized_ccm.sampler,
            pixel_images=poses,
            ip=ref_image,
            step=config.get("steps"),
            scale=config.get("cfg"),
            batch_size=self.initialized_ccm.batch_size,
            ddim_eta=0.0,
            dtype=self.initialized_ccm.dtype,
            device=self.initialized_ccm.device,
            camera=self.initialized_ccm.camera,
            num_frames=self.initialized_ccm.num_frames,
            pixel_control=(self.initialized_ccm.mode == "pixel"),
            transform=self.initialized_ccm.image_transform,
            offset_noise=self.initialized_ccm.offset_noise,
            callback=prog
        )

        return (torch.stack([
            torch.from_numpy(img / 255.0) for img in stage2_images
        ]),)

class CRMModelLoader:
    def __init__(self):
        self.initialized_model = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "crm_model": (get_filename_list("checkpoints"),),
            }
        }

    RETURN_TYPES = ("CRM_MODEL",)
    FUNCTION = "load"
    CATEGORY = "Flowty CRM"

    def load(self, crm_model):
        device = get_torch_device()

        if not torch.cuda.is_available():
            device = "cpu"

        if not self.initialized_model:
            print("Loading CRM model")
            specs = json.load(open(path.join(path.dirname(__file__), "crmlib/configs/specs_objaverse_total.json")))
            model = CRM(specs).to(device)
            model.load_state_dict(torch.load(get_full_path("checkpoints", crm_model), map_location=device),
                                  strict=False)
            self.initialized_model = model

        return (self.initialized_model,)


class CRMModeler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "crm_model": ("CRM_MODEL",),
                "poses": ("IMAGE",),
                "coordinates": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MESH",)
    FUNCTION = "make_model"
    CATEGORY = "Flowty CRM"

    def make_model(self, crm_model, poses, coordinates):
        device = get_torch_device()

        if not torch.cuda.is_available():
            device = "cpu"

        rgb_list = []
        ccm_list = []

        for img in poses:
            rgb_list.append(Image.fromarray(np.clip(255. * img.detach().cpu().numpy(), 0, 255).astype(np.uint8)))

        for img in coordinates:
            ccm_list.append(Image.fromarray(np.clip(255. * img.detach().cpu().numpy(), 0, 255).astype(np.uint8)))

        rgb = np.concatenate(rgb_list, 1)
        ccm = np.concatenate(ccm_list, 1)

        mesh = generate3d(crm_model, rgb, ccm, device)

        return ({"mesh": mesh, "cuda": False},)

class CRMModelerCuda:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "crm_model": ("CRM_MODEL",),
                "poses": ("IMAGE",),
                "coordinates": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MESH",)
    FUNCTION = "make_model"
    CATEGORY = "Flowty CRM"

    def make_model(self, crm_model, poses, coordinates):
        if not torch.cuda.is_available():
            raise Exception("Cuda is not supported, use the regular CRMModeler node")

        device = get_torch_device()

        rgb_list = []
        ccm_list = []

        for img in poses:
            rgb_list.append(Image.fromarray(np.clip(255. * img.detach().cpu().numpy(), 0, 255).astype(np.uint8)))

        for img in coordinates:
            ccm_list.append(Image.fromarray(np.clip(255. * img.detach().cpu().numpy(), 0, 255).astype(np.uint8)))

        rgb = np.concatenate(rgb_list, 1)
        ccm = np.concatenate(ccm_list, 1)

        mesh = generate3d_cuda(crm_model, rgb, ccm, device)

        return ({"mesh": mesh, "cuda": True},)

def do_resize_content(original_image: Image, scale_rate):
    # resize image content wile retain the original image size
    if scale_rate != 1:
        # Calculate the new size after rescaling
        new_size = tuple(int(dim * scale_rate) for dim in original_image.size)
        # Resize the image while maintaining the aspect ratio
        resized_image = original_image.resize(new_size)
        # Create a new image with the original size and black background
        padded_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
        paste_position = (
        (original_image.width - resized_image.width) // 2, (original_image.height - resized_image.height) // 2)
        padded_image.paste(resized_image, paste_position)
        return padded_image
    else:
        return original_image


def expand_to_square(image, bg_color=(0, 0, 0, 0)):
    # expand image to 1:1
    width, height = image.size
    if width == height:
        return image
    new_size = (max(width, height), max(width, height))
    new_image = Image.new("RGBA", new_size, bg_color)
    paste_position = ((new_size[0] - width) // 2, (new_size[1] - height) // 2)
    new_image.paste(image, paste_position)
    return new_image


def add_background(image, bg_color=(255, 255, 255)):
    # given an RGBA image, alpha channel is used as mask to add background color
    background = Image.new("RGBA", image.size, bg_color)
    return Image.alpha_composite(background, image)


class CRMPreprocessForPoser:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "reference_mask": ("MASK",),
                "foreground_ratio": ("FLOAT", {"default": 1, "min": 0.5, "max": 1, "step": 0.1})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed",)
    FUNCTION = "preprocess"
    CATEGORY = "Flowty CRM"

    def preprocess(self, reference_image, reference_mask, foreground_ratio):
        image = reference_image[0]
        mask = reference_mask[0].unsqueeze(2)
        image = torch.cat((image, mask), dim=2).detach().cpu().numpy()
        image = Image.fromarray(np.clip(255. * image, 0, 255).astype(np.uint8), "RGBA")
        image = do_resize_content(image, foreground_ratio)
        image = expand_to_square(image)
        image = add_background(image, "#7F7F7F").convert("RGB")
        return (torch.stack([torch.from_numpy(np.array(image).astype(np.uint8) / 255.0)]),)

class CRMViewer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH",)
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "display"
    CATEGORY = "Flowty CRM"

    def display(self, mesh):
        saved = list()
        full_output_folder, filename, counter, subfolder, filename_prefix = get_save_image_path("meshsave",
                                                                                                get_output_directory())

        for (batch_number, single_mesh) in enumerate([mesh.get("mesh")]):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            if mesh.get("cuda", False):
                file = f"{filename_with_batch_num}_{counter:05}_.glb"
                single_mesh.write(path.join(full_output_folder, file))
            else:
                file = f"{filename_with_batch_num}_{counter:05}_.obj"
                single_mesh.apply_transform(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))
                single_mesh.export(path.join(full_output_folder, file))
            saved.append({
                "filename": file,
                "type": "output",
                "subfolder": subfolder
            })

        return {"ui": {"mesh": saved}}


NODE_CLASS_MAPPINGS = {
    "CRMPreprocessForPoser": CRMPreprocessForPoser,
    "CCMSampler": CCMSampler,
    "CRMPoseSampler": CRMPoseSampler,
    "CRMPoserConfig": CRMPoserConfig,
    "CRMModelLoader": CRMModelLoader,
    "CRMModeler": CRMModeler,
    "CRMModelerCuda": CRMModelerCuda,
    "CRMViewer": CRMViewer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRMPreprocessForPoser": "CRM Preprocessor For Poser",
    "CCMSampler": "CCM Sampler",
    "CRMPoseSampler": "CRM Pose Sampler",
    "CRMPoserConfig": "CRM PoserConfig",
    "CRMModelLoader": "CRM Model Loader",
    "CRMModeler": "CRM Modeler",
    "CRMModelerCuda": "CRM Modeler (Cuda only)",
    "CRMViewer": "CRM Viewer",
}

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
