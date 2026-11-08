# VLM-LLM-UQ
## Uncertainty Quantification of Vision Language Models and Large Language Models

Demo -> [https://huggingface.co/spaces/aseembits93/VLM-LLM-UQ](Demo)

### Instructions

```
conda env create -f environment.yml
conda activate uq
pip install git+https://github.com/haotian-liu/LLaVA.git 
python app.py
```
Structure your prompt in the following way
```
Question
A. Option A
B. Option B
C. Option C
D. Option D
E. I don’t know
F. None of the above
```

## Contrastive retrieval idea

When the prediction has multiple answers, does appending
only those surviving answer texts to the retrieval query find the supporting
passage more often than querying with the question alone?

```bash
python -m experiments.contrastive_retrieval \
  --data mmbench.pkl \
  --output results/contrastive_retrieval_mvp.json \
  --plot results/contrastive_retrieval_mvp.png
```

APS candidate-based retrieval actually reduces accuracy.
