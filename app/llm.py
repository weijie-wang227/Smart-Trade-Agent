from dataclasses import dataclass
import os
import google.genai as genai
from google.genai import types



@dataclass
class GeminiLLM:
    model: str = "gemini-2.0-flash"
    temperature: float = 0.0
    GEMINI_INPUT_PRICE_PER_1M: float = 0.10
    GEMINI_OUTPUT_PRICE_PER_1M: float = 0.4

    def __post_init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")

        self.client = genai.Client(api_key=api_key)
        print(f"Initialized GeminiLLM with model={self.model}")

    def invoke(self, prompt: str) -> str:
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
            )
        )

        usage = response.usage_metadata or {}     
#

        return response.text.strip(), self.gemini_cost_from_usage(usage)
    
    def gemini_cost_from_usage(self, usage: dict) -> float:
        if usage is None:
            return 0.0
        
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0
        
        cost = (
            (prompt_tokens / 1000000) * self.GEMINI_INPUT_PRICE_PER_1M
            + (output_tokens / 1000000) * self.GEMINI_OUTPUT_PRICE_PER_1M
        )
        return cost
    

