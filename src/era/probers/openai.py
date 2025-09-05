import os
from typing import List, Dict

class OpenAIProber:
    def __init__(self, model_name: str, base_url: str = None, api_key: str = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=base_url)
        self.model = model_name
    def generate(self, prompt: str, max_new_tokens: int = 60) -> str:
        r = self.client.chat.completions.create(model=self.model, messages=[{"role":"user","content":prompt}], max_tokens=max_new_tokens, temperature=0.2)
        return r.choices[0].message.content
    def candidate_logprobs(self, prompt: str, candidates: List[str]) -> Dict[str, float]:
        # Fallback implementation: returns zeros if logprobs unsupported.
        return {c: 0.0 for c in candidates}
