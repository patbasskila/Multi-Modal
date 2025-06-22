import json
from io import BytesIO
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from router import handle_request
from utils import bytes_to_b64

app = FastAPI()

# Enable CORS so your React app (http://localhost:3000) can call /api/query directly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    modality: str
    payload: str

@app.post("/api/query")
async def query(q: Query):
    """
    Receive JSON:
      { "modality": "text" | "audio" | "image",
        "payload":  "<base64>" or "<text>" }
    """
    res = handle_request(q.modality, q.payload)

    # Encode any raw audio bytes to base64
    if "audio_bytes" in res:
        res["confirm_audio"] = bytes_to_b64(res.pop("audio_bytes"), "audio/wav")

    # Encode any PIL.Image annotations to base64 PNG
    if "annotated_image" in res:
        img = res.pop("annotated_image")
        buf = BytesIO()
        img.save(buf, format="PNG")
        res["annotated_image"] = bytes_to_b64(buf.getvalue(), "image/png")

    # Encode any raw generated image bytes to base64 PNG
    if "image_bytes" in res:
        res["generated_image"] = bytes_to_b64(res.pop("image_bytes"), "image/png")

    return res

# @app.websocket("/ws/trace")
# async def ws_trace(ws: WebSocket):
#     """
#     Establish a WebSocket that streams the current SVG of the StateGraph
#     each time any node is executed. The frontend’s GraphViewer will render it.
#     """
#     await ws.accept()

#     def trace_fn(state):
#         # graph.visualize() returns an SVG string representing the current state
#         svg = graph.visualize()
#         payload = json.dumps({"svg": svg})
#         import asyncio
#         # Send asynchronously so we don’t block the event loop
#         asyncio.create_task(ws.send_text(payload))

#     # Register our callback
#     graph.set_tracer(trace_fn)

#     # Keep the connection open, ignoring incoming messages
#     while True:
#         await ws.receive_text()
