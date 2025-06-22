import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.schema import Document
from utils import web_search


class LLMAgent:
    def __init__(self, model_name: str = None, quantize: bool = True):
        """
        LLMAgent loads an open-source LLaMA variant and falls back on web search when confidence is low.

        - model_name: Hugging Face repo ID for the LLaMA model.
        - quantize: Whether to load the model in 8-bit quantized mode (requires bitsandbytes and CUDA).
        """
        # 1) Choose default model if none provided
        if not model_name:
            model_name = "meta-llama/Llama-2-7b-chat-hf"
            print("LLMAgent: No model_name provided. Defaulting to 'meta-llama/Llama-2-7b-chat-hf'.")

        print(f"LLMAgent: Attempting to load '{model_name}' …")

        # 2) Detect device (GPU vs. CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"LLMAgent: Using device = {self.device}")

        # 3) Load tokenizer with authentication token if needed
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_auth_token=True
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load tokenizer for '{model_name}'.\n"
                "Ensure the model ID is correct and public (or that you have proper auth)."
                f"\nOriginal error:\n{e}"
            ) from e

        # Ensure we have a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 4) Prepare model loading kwargs
        model_kwargs = {}
        if torch.cuda.is_available() and quantize:
            print("LLMAgent: CUDA detected → loading model in 8-bit mode with auto device map.")
            model_kwargs.update({
                "load_in_8bit": True,
                "device_map": "auto",
                "llm_int8_enable_fp32_cpu_offload": True,
            })
        elif torch.cuda.is_available():
            print("LLMAgent: CUDA detected → loading model in FP16 mode on GPU.")
            model_kwargs.update({
                "device_map": "auto",
                "torch_dtype": torch.float16,
            })
        else:
            print("LLMAgent: No CUDA detected → loading model in full precision on CPU.")
            model_kwargs.update({
                "device_map": {"": "cpu"},
                "torch_dtype": torch.float32,
                "low_cpu_mem_usage": True,
            })

        # 5) Load model with authentication, then move to device
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                use_auth_token=True,
                **model_kwargs
            )
            self.model.eval()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model weights for '{model_name}'.\n"
                "If you see a bitsandbytes error on CPU, it means the code tried 8-bit without CUDA.\n"
                "Make sure you have a GPU (and bitsandbytes installed) if you want 8-bit, "
                "or switch to a smaller CPU-friendly model.\n"
                f"Original error:\n{e}"
            ) from e

        print(f"LLMAgent: Successfully loaded '{model_name}'.\n")

    def run(self, prompt: str, max_new_tokens: int = 256) -> dict:
        """
        1. Perform an initial generation on `prompt`.
        2. If "confidence" < 0.7, perform a web search and regenerate using the results.

        Returns a dict:
            {
                "text":       <generated text>,
                "confidence": <float 0.0–1.0>,
                "task":       <"text" | "image_gen" | "vision">
            }
        """
        print(f"🔍 [LLMAgent.run] prompt = {prompt!r}")

        def detect_task(generated_text: str) -> str:
            s = generated_text.lower()
            if "generate image" in s:
                return "image_gen"
            if "describe" in s:
                return "vision"
            return "text"

        # 1) INITIAL GENERATION
        instruction = f"### Instruction:\n{prompt}\n### Response:"
        inputs = self.tokenizer(instruction, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.95,
                top_k=50,
            )
        resp = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Compute a simple confidence metric
        conf = min(len(resp) / 100, 1.0)
        task = detect_task(resp)

        # 2) LOW-CONFIDENCE FALLBACK
        if conf < 0.7:
            docs: list[Document] = web_search(prompt, num_results=3)
            docs_text = "\n".join(f"- {doc.page_content}" for doc in docs)

            fallback_prompt = (
                f"{prompt}\n\n"
                f"Web search results:\n{docs_text}\n\n"
                "Based on the above, please answer:"
            )
            inputs2 = self.tokenizer(fallback_prompt, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs2 = self.model.generate(
                    **inputs2,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                )
            resp2 = self.tokenizer.decode(outputs2[0], skip_special_tokens=True)

            conf2 = min(len(resp2) / 100, 1.0)
            task2 = detect_task(resp2)

            if resp2.strip():
                resp, conf, task = resp2, conf2, task2

        print(f"🔍 [LLMAgent.run] returning: {{'text': {resp!r}, 'confidence': {conf}, 'task': {task!r}}}")
        return {"text": resp, "confidence": conf, "task": task}
