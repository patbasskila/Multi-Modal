import os
import re
from typing import TypedDict, Any, List, Dict
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.base import BaseCheckpointSaver

from agents.asr_agent    import ASRAgent
from agents.tts_agent    import TTSAgent
from agents.llm_agent    import LLMAgent
from agents.vision_agent import VisionAgent
from agents.imggen_agent import ImageGenAgent


class NoopCheckpoint(BaseCheckpointSaver):
    # streaming code checks for this attribute
    config_specs = []

    def save(self, state: dict, name: str, **kwargs) -> None:
        # no‐op
        return None

    def load(self, name: str, **kwargs) -> dict:
        # nothing to restore
        return {}
    

# --- Define the shared state schema for the graph ---
class AppState(TypedDict, total=False):
    text: str
    audio_bytes: bytes
    confidence: float
    task: str
    objects: List[Dict[str, Any]]
    annotated_image: Any
    image_bytes: bytes

# --- Paths & Config ---
BASE_DIR       = os.path.dirname(__file__)
CONFIG_DIR     = os.path.join(BASE_DIR, "config")
LABEL_MAP_JSON = os.path.join(CONFIG_DIR, "label_map.json")

# --- Initialize each “agent” ---
asr_agent    = ASRAgent()
tts_agent    = TTSAgent()
llm_agent    = LLMAgent()
vision_agent = VisionAgent(
    label_map_path       = LABEL_MAP_JSON,
    confidence_threshold = 0.5
)
imggen_agent = ImageGenAgent()

# --- Build the StateGraph Orchestrator with schema ---
graph = StateGraph(AppState)

def tts_adapter(state: AppState) -> AppState:
    # take the transcript, run TTS, then return only the new audio_bytes
    tts_out = tts_agent.run(state["text"])
    return {"audio_bytes": tts_out["audio_bytes"]}

graph.add_node("tts", RunnableLambda(tts_adapter))


# def llm_adapter(state: AppState) -> AppState:
#     # take the text (original or post-ASR), run LLM, then return its outputs
#     llm_out = llm_agent.run(state["text"])
    
#     # Clean up the raw generated text by removing the template prefix
#     text = llm_out["text"]
#     # Look for our marker and cut everything before it
#     marker = "### Response:"
#     if marker in text:
#         # keep everything after '### Response:\n'
#         text = text.split(marker, 1)[1].lstrip("\n ")
#     # Now text is just the model’s reply
#     llm_out["text"] = text
    
#     # llm_out is {"text":..., "confidence":..., "task":...}
#     return llm_out

def llm_adapter(state: AppState) -> AppState:
    # 1) Grab the prompt that the graph is about to send to the LLM
    prompt = state.get("text")
    print(f"\n[llm_adapter] 🔍 Prompt to LLMAgent.run: {prompt!r}")

    # 2) Call the agent
    llm_out = llm_agent.run(prompt)

    # 3) Extract the first response via regex
    full_text = llm_out["text"]
    match = re.search(
        r"### Response:\s*([\s\S]*?)(?=### Instruction:|$)",
        full_text,
        flags=re.IGNORECASE,
    )
    if match:
        cleaned = match.group(1).strip()
    else:
        # Fallback: strip all markers if regex fails
        cleaned = re.sub(r"### (Instruction|Response):", "", full_text, flags=re.IGNORECASE).strip()
    
    # 4) Overwrite only the text field
    llm_out["text"] = cleaned
    
    return llm_out

# def llm_adapter(state: AppState) -> AppState:
#     prompt = state.get("text")
#     print(f"🔍 [llm_adapter] received prompt: {prompt!r}")
#     llm_out = llm_agent.run(prompt)
#     print(f"🔍 [llm_adapter] llm_out = {llm_out!r}")
#     return llm_out


graph.add_node("llm", RunnableLambda(llm_adapter))
# Add nodes (each node’s .run(...) will be invoked)
graph.add_node("asr",    RunnableLambda(asr_agent.run))      # .run(audio_bytes) -> { "text":..., "confidence":... }
# graph.add_node("tts",    RunnableLambda(tts_agent.run))      # .run(text)       -> { "audio_bytes":... }
# graph.add_node("llm",    RunnableLambda(llm_agent.run))      # .run(text)       -> { "text":..., "confidence":..., "task":... }
graph.add_node("vision", RunnableLambda(vision_agent.run))   # .run(image)      -> { "objects":..., "annotated_image": PIL.Image }
graph.add_node("imggen", RunnableLambda(imggen_agent.run))   # .run(prompt)     -> { "image_bytes":... }


# --- Define routing edges ---

# Add edges from START to possible entry points
# graph.add_edge(START, "asr")
# graph.add_edge(START, "llm")

# 1) Entrypoint routing from implicit START → ASR or LLM
def start_router(state: AppState) -> str:
    """
    Routes to the appropriate entry point based on input state
    """
    if state.get("audio_bytes") is not None:
        return "asr"
    elif state.get("text") is not None:
        return "llm"
    else:
        raise ValueError("No valid input provided to route from START")

graph.add_conditional_edges(
    START,
    start_router,
    {
      "asr": "asr",
      "llm": "llm",
      None: END   
    }
)

# 2) AUDIO → ASR → TTS → LLM
graph.add_edge("asr", "tts")   # ASRAgent.run → returns "text"
graph.add_edge("tts", "llm")   # TTSAgent.run → returns "audio_bytes" and forwards "text" to LLM

# 3) Conditional routing from LLM based on task
def llm_router(state: AppState) -> str:
    task = state.get("task")
    if task == "vision":
        return "vision"
    elif task == "image_gen":
        return "imggen"
    else:
        return None  # Stops if no matching task

graph.add_conditional_edges(
    "llm", 
    llm_router,
    {
        "vision": "vision",
        "imggen": "imggen",
        None: END  # If no task matches, end the flow
    }  
)
graph.add_edge("vision", END)  # Vision processing ends the flow
graph.add_edge("imggen", END)  # Image generation also ends the flow


# Compile the graph, passing the start_router
app_flow = graph.compile(checkpointer=False)

# --- Define dynamic entry point routing logic (optional) ---
def run_graph(input_state: AppState):
    """
    Runs the compiled app_flow graph from a dynamic entry point,
    depending on the available input fields.
    """
    return app_flow.invoke(input_state)  # Let the graph handle the start

