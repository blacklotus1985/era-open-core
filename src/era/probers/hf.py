import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def _sum_token_logprobs(logits_last, tokenizer, candidate: str) -> float:
    tokens = tokenizer.encode(candidate, add_special_tokens=False)
    log_probs = torch.log_softmax(logits_last, dim=-1)
    return float(sum(log_probs[tid].item() for tid in tokens))

class HFProber:
    def __init__(self, model_name: str, revision: str = None):
        self.tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision)
        self.model.eval()
    @torch.no_grad()
    def last_logits(self, prompt: str):
        inp = self.tok(prompt, return_tensors="pt")
        out = self.model(**inp)
        return out.logits[0, -1, :]
    @torch.no_grad()
    def candidate_logprobs(self, prompt: str, candidates):
        logits = self.last_logits(prompt)
        return {c: _sum_token_logprobs(logits, self.tok, c) for c in candidates}
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 40):
        inp = self.tok(prompt, return_tensors="pt")
        gen = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.2)
        return self.tok.decode(gen[0], skip_special_tokens=True)
