# Attention Mechanism – A Gentle Introduction

## 🌟 What You’ll Learn

- How words are represented as vectors (embeddings)
- The role of **Queries (Q)**, **Keys (K)**, and **Values (V)**
- How dot-product attention is computed step by step
- Why scaling and softmax are applied
- The final attention-weighted output

## 📝 Step-by-Step Summary

1. **Word Embeddings** – Convert words ("The", "cat", "sat") into numeric vectors.
2. **Q, K, V Matrices** – Apply linear transformations to get Queries, Keys, and Values.
3. **Dot Products** – Compare Queries with Keys to measure relevance.
4. **Scaling** – Normalize scores by dividing by √dₖ to prevent large values.
5. **Softmax** – Turn scores into probabilities (attention weights).
6. **Weighted Sum** – Multiply weights by Values → this gives the context-aware representation.

## 💻 Code Highlight

The notebook demonstrates attention in just a few lines of NumPy:

```python
scores = Q @ K.T              # similarity
scaled = scores / np.sqrt(dk) # scaling
weights = softmax(scaled)     # attention weights
output = weights @ V          # final output
```

## 🚀 Key Takeaways

- Attention tells the model *where to look* in a sequence.
- Queries ask a question, Keys provide possible matches, Values supply the actual information.
- This mechanism is the foundation of **Transformers, BERT, and GPT**.

---

✨ Explore the notebook in this repo to see the math come alive with code!

