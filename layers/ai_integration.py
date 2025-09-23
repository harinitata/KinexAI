# layers/ai_integration.py
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

class AIIntegration:
    def __init__(self):
        """
        Initializes the AI model and tokenizer.
        This can take a few minutes on the first run as it downloads the model.
        """
        print("Loading AI model... This may take a moment.")
        # Check if a GPU is available, otherwise use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load the pre-trained FLAN-T5 model and tokenizer
        model_name = "google/flan-t5-base"
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        print("AI model loaded successfully.")

    # In layers/ai_integration.py, replace the get_ai_feedback method with this one.

    def get_ai_feedback(self, payload):
        """
        Takes the JSON payload, formats the feedback, and rephrases it using the AI model.
        """
        form_feedback = payload.get("form_feedback", {})
        
        if not form_feedback:
            return "Great form! Keep up the amazing work. Ready for the next set?"

        # Format the feedback points as a clear, bulleted list for the AI.
        corrections_list = "\n".join([f"- {value}" for value in form_feedback.values()])

        # --- NEW, MORE EFFECTIVE PROMPT ---
        # This structured format helps the model understand its task much better.
        prompt = f"""
Task: Rephrase the following technical fitness corrections into a single, friendly, and encouraging coaching tip.

Corrections:
{corrections_list}

Coaching Tip:
"""
        # --- END OF NEW PROMPT ---

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_length=60, num_beams=5, early_stopping=True)
        
        friendly_feedback = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        
        return friendly_feedback