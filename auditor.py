import torch
import time # Added for timing
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import os

model_id = "vikhyatk/moondream2"
revision = "2024-08-26" 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Stable Moondream Auditor on: {device.upper()}")

tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True,
    revision=revision,
    torch_dtype=torch.float16, 
    attn_implementation="eager",
    low_cpu_mem_usage=True
).to(device).eval()

def vlm_auditor(image_path: str, exercise_type: str):
    try:
        if not os.path.exists(image_path):
            return {"feedback": "Image file missing", "prompt_tokens": 0, "generated_tokens": 0, "ttft_sec": 0.0}

        image = Image.open(image_path).convert("RGB")
        image = image.resize((756, 756))
        
        prompt = (
            f"You are a strict fitness coach analyzing a {exercise_type}. "
            f"Do not use the word 'Perfect'. "
            f"First, briefly describe the position of the person's back and limbs. "
            f"Second, give your critique: If the form is flawed, say 'Form is wrong, ' followed by the specific fix. "
            f"If the form is completely flawless, say 'Keep it up! Good form.'"
        )
        
        # 1. Encode Image & Prompt
        image_embeds = model.encode_image(image)
        # We need to tokenize the prompt manually to count prompt tokens
        prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_tokens = prompt_inputs.input_ids.shape[1]

        # 2. Timing and Generation
        start_time = time.perf_counter()
        
        # Moondream's answer_question doesn't easily expose the raw generated tokens, 
        # so we get the string, then re-tokenize it to count.
        result = model.answer_question(image_embeds, prompt, tokenizer, max_new_tokens=128)
        
        ttft_sec = time.perf_counter() - start_time # Approximate TTFT

        answer = str(result).strip()
        
        # Count generated tokens
        generated_token_ids = tokenizer(answer, return_tensors="pt").input_ids
        generated_tokens = generated_token_ids.shape[1]

        return {
            "feedback": answer, 
            "prompt_tokens": prompt_tokens, 
            "generated_tokens": generated_tokens,
            "ttft_sec": ttft_sec
        }

    except Exception as e:
        print(f"AUDITOR ERROR: {e}")
        return {"feedback": f"Error: {str(e)}", "prompt_tokens": 0, "generated_tokens": 0, "ttft_sec": 0.0}