import os
import base64
from io import BytesIO
from PIL import Image
from serpapi import GoogleSearch

def b64_to_bytes(data_uri: str) -> bytes:
    """
    Decode a base64 data URI into raw bytes.
    """
    header, b64data = data_uri.split(",", 1)
    return base64.b64decode(b64data)


def bytes_to_b64(b: bytes, mime: str) -> str:
    """
    Encode raw bytes into a base64 data URI with the given MIME type.
    """
    return f"data:{mime};base64," + base64.b64encode(b).decode()


def load_image(b64: str) -> Image.Image:
    """
    Load a PIL.Image from a base64 data URI.
    """
    return Image.open(BytesIO(b64_to_bytes(b64))).convert("RGB")


def pil_to_b64(image: Image.Image, format: str = "PNG") -> str:
    """
    Convert a PIL.Image into a base64 data URI.
    """
    buf = BytesIO()
    image.save(buf, format=format)
    mime = f"image/{format.lower()}"
    return bytes_to_b64(buf.getvalue(), mime)


def web_search(query: str, num_results: int = 3) -> str:
    """
    Perform a real web search via SerpAPI and return a combined text summary.
    Requires SERPAPI_API_KEY in env.
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise EnvironmentError("SERPAPI_API_KEY not set in environment")

    params = {
        "engine": "google",
        "q": query,
        "num": num_results,
        "api_key": api_key
    }
    search = GoogleSearch(params)
    results = search.get_dict().get("organic_results", [])

    if not results:
        return f"No search results found for '{query}'."

    # Build a simple text summary
    summary_lines = []
    for idx, item in enumerate(results, start=1):
        title = item.get("title", "").strip()
        snippet = item.get("snippet", "").strip()
        summary_lines.append(f"{idx}. {title}: {snippet}")

    return "\n".join(summary_lines)
