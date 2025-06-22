from models import app_flow
from utils import b64_to_bytes, load_image
from typing import Dict, Any
from agents.llm_agent import LLMAgent


_llm_direct = LLMAgent()

def handle_request(modality: str, payload: str) -> Dict[str, Any]:
    """
    Given:
      - modality: one of "audio", "text", "image"
      - payload:   base64‐encoded string for audio/image, or raw text
    We either call the LLM directly (for text) or seed the StateGraph.
    """
    # --- TEXT: bypass the graph entirely for direct LLM output ---
    if modality == "text":
        # Returns a dict: {"text": ..., "confidence": ..., "task": ...}
        return _llm_direct.run(payload)

    # --- AUDIO or IMAGE go through the graph ---
    state: Dict[str, Any] = {}
    if modality == "audio":
        # Decode base64 → bytes for ASR
        state["audio_bytes"] = b64_to_bytes(payload)

    elif modality == "image":
        # Decode base64 → PIL.Image for vision
        state["image"] = load_image(payload)

    else:
        return {"error": True, "message": f"Unsupported modality: {modality}"}

    # Run the graph end-to-end and grab the last state snapshot
    final_state = None
    for snapshot in app_flow.invoke(state):
        final_state = snapshot

    if final_state is None or not isinstance(final_state, dict):
        return {"error": True, "message": "No output from flow or unexpected type."}

    return final_state