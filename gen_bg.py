import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS

def load_models():
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name="theAllysMixXSDXL_v10.safetensors"
        )

        emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
        emptylatentimage_5 = emptylatentimage.generate(
            width=1024, height=1024, batch_size=1
        )

        loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
        loraloader_15 = loraloader.load_lora(
            lora_name="lcm_lora_sdxl.safetensors",
            strength_model=1,
            strength_clip=1,
            model=get_value_at_index(checkpointloadersimple_4, 0),
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        modelsamplingdiscrete = NODE_CLASS_MAPPINGS["ModelSamplingDiscrete"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        return loraloader_15, emptylatentimage_5,checkpointloadersimple_4, cliptextencode, modelsamplingdiscrete, ksampler, vaedecode, saveimage

def gen_pic(positive_prompt,loraloader_15, emptylatentimage_5,checkpointloadersimple_4,cliptextencode, modelsamplingdiscrete, ksampler, vaedecode, saveimage):
    with torch.inference_mode():
        # cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_6 = cliptextencode.encode(
            text=positive_prompt,
            clip=get_value_at_index(loraloader_15, 1),
        )

        cliptextencode_7 = cliptextencode.encode(
            text="no modern technology, no futuristic elements, no neon lights, no contemporary furniture, no plastic, no vehicles, no bright daylight, no sci-fi details, no electronic devices, no modern bar items, no overly clean or polished surfaces, no smooth metal, no characters, no modern drinks or glassware, no cityscape, no overly bright or colorful elements, no clutter or random objects, no modern clothing or accessories.",
            clip=get_value_at_index(loraloader_15, 1),
        )

        # modelsamplingdiscrete = NODE_CLASS_MAPPINGS["ModelSamplingDiscrete"]()
        # ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        # vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        # saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        for q in range(1):
            modelsamplingdiscrete_16 = modelsamplingdiscrete.patch(
                sampling="eps", zsnr=False, model=get_value_at_index(loraloader_15, 0)
            )

            ksampler_3 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=4,
                cfg=1,
                sampler_name="lcm",
                scheduler="sgm_uniform",
                denoise=1,
                model=get_value_at_index(modelsamplingdiscrete_16, 0),
                positive=get_value_at_index(cliptextencode_6, 0),
                negative=get_value_at_index(cliptextencode_7, 0),
                latent_image=get_value_at_index(emptylatentimage_5, 0),
            )

            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(ksampler_3, 0),
                vae=get_value_at_index(checkpointloadersimple_4, 2),
            )

            saveimage_23 = saveimage.save_images(
                filename_prefix="ComfyUI", images=get_value_at_index(vaedecode_8, 0)
            )

def main():
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name="theAllysMixXSDXL_v10.safetensors"
        )

        emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
        emptylatentimage_5 = emptylatentimage.generate(
            width=1024, height=1024, batch_size=1
        )

        loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
        loraloader_15 = loraloader.load_lora(
            lora_name="lcm_lora_sdxl.safetensors",
            strength_model=1,
            strength_clip=1,
            model=get_value_at_index(checkpointloadersimple_4, 0),
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_6 = cliptextencode.encode(
            text="A secret pirate hideout nestled within a rocky cove, with ancient ships anchored by the shore, their sails torn and weathered. The cove is surrounded by jagged cliffs and dark storm clouds brewing overhead, with flashes of lightning illuminating the scene. Wooden docks lead to a bustling tavern built into the cliffs, and treasure chests filled with gold and jewels are scattered across the beach. The air smells of saltwater and danger.",
            clip=get_value_at_index(loraloader_15, 1),
        )

        cliptextencode_7 = cliptextencode.encode(
            text="no modern technology, no futuristic elements, no neon lights, no contemporary furniture, no plastic, no vehicles, no bright daylight, no sci-fi details, no electronic devices, no modern bar items, no overly clean or polished surfaces, no smooth metal, no characters, no modern drinks or glassware, no cityscape, no overly bright or colorful elements, no clutter or random objects, no modern clothing or accessories.",
            clip=get_value_at_index(loraloader_15, 1),
        )

        modelsamplingdiscrete = NODE_CLASS_MAPPINGS["ModelSamplingDiscrete"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        for q in range(1):
            modelsamplingdiscrete_16 = modelsamplingdiscrete.patch(
                sampling="eps", zsnr=False, model=get_value_at_index(loraloader_15, 0)
            )

            ksampler_3 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=4,
                cfg=1,
                sampler_name="lcm",
                scheduler="sgm_uniform",
                denoise=1,
                model=get_value_at_index(modelsamplingdiscrete_16, 0),
                positive=get_value_at_index(cliptextencode_6, 0),
                negative=get_value_at_index(cliptextencode_7, 0),
                latent_image=get_value_at_index(emptylatentimage_5, 0),
            )

            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(ksampler_3, 0),
                vae=get_value_at_index(checkpointloadersimple_4, 2),
            )

            saveimage_23 = saveimage.save_images(
                filename_prefix="ComfyUI", images=get_value_at_index(vaedecode_8, 0)
            )


if __name__ == "__main__":
    main()
