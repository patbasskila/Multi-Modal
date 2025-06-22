import torch
from io import BytesIO
from diffusers import StableDiffusionPipeline
from PIL import Image


class ImageGenAgent:
    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5"):
        """
        Loads a Stable Diffusion pipeline for text-to-image generation.
        
        Args:
            model_name: Hugging Face repo ID of a public SD checkpoint.
                        Defaults to "runwayml/stable-diffusion-v1-5".
        """
        # Determine device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ImageGenAgent: Loading '{model_name}' onto {self.device}…")

        # Load the Stable Diffusion pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        ).to(self.device)

        # Disable the safety checker to avoid extra dependencies/issues
        self.pipe.safety_checker = lambda images, **kwargs: (images, False)

    def run(self, prompt: str) -> dict:
        """
        Generates an image for the given prompt and returns raw PNG bytes.
        
        Args:
            prompt: Text prompt for image generation.
        
        Returns:
            {
                "image_bytes": <PNG bytes>
            }
        """
        # Generate a single image (1 inference step)
        with torch.autocast("cuda") if self.device.type == "cuda" else torch.no_grad():
            image: Image.Image = self.pipe(prompt).images[0]

        # Convert PIL Image to PNG bytes
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()

        return {"image_bytes": img_bytes}

