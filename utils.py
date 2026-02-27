from PIL import Image
import base64, torch
from io import BytesIO

def get_b64_png_str(image_object: Image.Image) -> str:
    """
    Converts a PIL Image object to a base64 encoded PNG string.
    """
    # Create an in-memory buffer
    buffered = BytesIO()

    # Save the image to the buffer in PNG format
    image_object.save(buffered, format="PNG")

    # Get the binary value from the buffer
    img_bytes = buffered.getvalue()

    # Encode the bytes to base64
    img_base64_bytes = base64.b64encode(img_bytes)

    # Decode the bytes to a UTF-8 string for common use cases
    img_base64_string = img_base64_bytes.decode('utf-8')

    return img_base64_string

def get_torch_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def get_url_base64_str(image_object: Image.Image) -> str:
    img_base64_string = get_b64_png_str(image_object)
    return f"data:image/png;base64,{img_base64_string}"
