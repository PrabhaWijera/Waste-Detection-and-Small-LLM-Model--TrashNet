# app/models/waste_llm.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class WasteLLM:
    def __init__(self, model_name="EleutherAI/gpt-neo-125M", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def recommend(self, text: str, region=None, city=None) -> str:
        text = text.strip()
        if not text:
            return "- Dispose of waste according to local guidelines."

        # You can ignore region/city in prompt
        prompt = f"""
    You are an AI assistant giving short, clear, and humanized tips for proper disposal or recycling of specific waste items.
    Examples:
    Waste: "plastic bottle"
    Tips:
    - Rinse and dry the bottle.
    - Recycle in designated plastic bins.
    - Keep the cap on if required.
    - Never burn plastics.

    Waste: "cardboard box"
    Tips:
    - Flatten the box to save space.
    - Tie bundles with twine.
    - Keep away from moisture.
    - Recycle at designated collection points.

    Waste: "{text}"
    Tips:
    """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.2,
                top_k=50,
                top_p=0.9,
                repetition_penalty=2.0,
                pad_token_id=self.tokenizer.eos_token_id
            )
            recommendation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            recommendation = recommendation.replace(prompt, "").strip()

            # Ensure proper bullet formatting
            lines = [line.strip("- ").strip() for line in recommendation.split("\n") if line.strip()]
            recommendation = "\n".join(f"- {line}" for line in lines if line)

            return recommendation

        except Exception as e:
            print(f"[WasteLLM] Model error: {e}")
            return "- Dispose of waste according to local guidelines."
