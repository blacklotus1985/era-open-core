import os
from typing import List, Dict

class AnthropicProber:
    def __init__(self, model_name: str, api_key: str = None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model_name
    def generate(self, prompt: str, max_new_tokens: int = 60) -> str:
        msg = self.client.messages.create(model=self.model, max_tokens=max_new_tokens, temperature=0.2, messages=[{"role":"user","content":prompt}])
        return "".join([c.text for c in msg.content if getattr(c, "text", None)])
    def candidate_logprobs(self, prompt: str, candidates: List[str]) -> Dict[str, float]:
        return {c: 0.0 for c in candidates}
