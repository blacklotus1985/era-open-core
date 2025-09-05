# ERA Methodology (Formal)

- Vocabulary Ω, embeddings E_M, scoring f_M, state update h_{t,M}.
- Embedding drift: d_emb = 1 - cos(mean(E(out_M)), mean(E(out_M'))).
- Probability shift: build P_c, Q_c from last-step log-probs over candidate set S_c; use symmetric KL + W1.
- CSI: w_emb * (d_emb/(1+d_emb)) + w_prob * (d_prob/(1+d_prob)); bounds in [0,1).

Controls: sentinel concepts, placebo, bootstrap CI, BH-FDR, change-point detection.
