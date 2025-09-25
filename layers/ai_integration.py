# layers/ai_integration.py
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

class AIIntegration:
    def __init__(self):
        print("Loading AI model... This may take a moment.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        model_name = "google/flan-t5-base"
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        print("AI model loaded successfully.")

    def rephrase_feedback(self, technical_feedback):
        """
        Takes a string of technical feedback and rephrases it.
        """
        if not technical_feedback:
            return "Great form on that last rep! Keep it up."

        prompt = f"""
Task: Rephrase the following technical fitness corrections into a single, friendly, and encouraging coaching tip.

Corrections:
- {technical_feedback}

Coaching Tip:
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_length=60, num_beams=5, early_stopping=True)
        
        friendly_feedback = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return friendly_feedback