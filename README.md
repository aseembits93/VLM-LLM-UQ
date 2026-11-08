# VLM-LLM-UQ
## Uncertainty Quantification of Vision Language Models and Large Language Models

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

## Contrastive retrieval MVP

When the prediction has multiple answers, does appending
only those surviving answer texts to the retrieval query find the supporting
passage more often than querying with the question alone?

It treats the 97 unique MMBench supporting passages as the retrieval corpus.

```bash
python -m experiments.contrastive_retrieval \
  --data mmbench.pkl \
  --output results/contrastive_retrieval_mvp.json \
  --plot results/contrastive_retrieval_mvp.png
```

On the 142 held-out data points, the contrastive query improved
Hit@5 from 81.0% to 89.4%.

![Contrastive retrieval experiment results](results/contrastive_retrieval_mvp.png)

### Results with/without APS candidate-based retrieval

| Metric | Primary: 114 group representatives | Secondary: 460 records including rotations |
| --- | ---: | ---: |
| Final accuracy, draft-based retrieval | 85/114 = 74.56% | 322/460 = 70.00% |
| Final accuracy, APS candidate retrieval | 84/114 = 73.68% | 316/460 = 68.70% |
