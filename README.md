# Awesome-llm-as-judges
[![GitHub Sponsors](https://img.shields.io/badge/sponsors-GitHub-blue?logo=github&logoColor=white)](https://github.com/sponsors) ![Build Status](https://img.shields.io/badge/build-passing-brightgreen) ![License](https://img.shields.io/badge/license-MIT-yellow) ![Contributors](https://img.shields.io/badge/contributors-10-yellow) ![Awesome List](https://img.shields.io/badge/awesome-awesome-brightgreen) ![](https://img.shields.io/badge/PRs-Welcome-red)


## Awesome-llm-as-judges: A Survey
This repo include the papers discussed in our latest survey paper on Awesome-llm-as-judges.

🔥: Read the full paper here: [Paper Link](xxx)

## Reference
If our survey is useful for your research, please kindly cite our [paper](https://arxiv.org/abs/2411.16594):
```
在这里添加引用！
```

## Overview of Awesome-LLM-as-a-judge:
![Overview](./img/overview.png)
![limit](./img/limit.png)

# 1. Functionality

## 1.1 Performance Evaluation
### 1.1.1 Responses Evaluation
### 1.1.2 Model Evaluation

## 1.2 Model Enhancement
### 1.2.1 Reward Modeling During Training
### 1.2.2 Acting as Verifier During Inference
### 1.2.3 Feedback for Refinement

## 1.3 Data Collection
### 1.3.1 Data Annotation
### 1.3.2 Data Synthesize

# 2. METHODOLOGY
## 2.1 Single-LLM System
### 2.1.1 Prompt-based
#### 2.1.1.1 In-Context Learning
#### 2.1.1.2 Step-by-step
#### 2.1.1.3 Definition Augmentation
#### 2.1.1.4 Multi-turn Optimization

### 2.1.2 Tuning-based
### 2.1.2.1 Score-based Tuning
### 2.1.2.2 Preference-based Learning

### 2.1.3 Post-processing
### 2.1.3.1 Probability Calibration
### 2.1.3.2 Text Reprocessing

## 2.2 Multi-LLM System
### 2.2.1 Communication
#### 2.2.1.1 Cooperation
#### 2.2.1.2 Competition

### 2.2.2 Aggregation

## 2.3 Hybrid System

# 3. APPLICATION
## 3.1 General
## 3.2 Multimodal
## 3.3 Medical
## 3.4 Legal
## 3.5 Financial
## 3.6 Education
## 3.7 Information Retrieval
## 3.8 Others

# 4. META-EVALUATION
## 4.1 Benchmarks
### 4.1.1 Code Generation
### 4.1.2 Machine Translation
### 4.1.3 Text Summarization
### 4.1.4 Dialogue Generation
### 4.1.5 Automatic Story Generation
### 4.1.6 Values Alignment
### 4.1.7 Recommendation
### 4.1.8 Search
### 4.1.9 Comprehensive Data
## 4.2 Metric


# 5. LIMITATION
## 5.1 Biases
### 5.1.1 Presentation-Related Biases
- Large language models are not robust multiple choice selectors
  ICLR 2024 [**Paper**](https://openreview.net/pdf?id=shr9PXz7T0)
- Look at the first sentence: Position bias in question answering
  EMNLP 2020 [**Paper**](https://aclanthology.org/2020.emnlp-main.84/)
- Batch calibration: Rethinking calibration for in-context learning and prompt engineering
  ICLR 2024 [**Paper**](https://openreview.net/pdf?id=L3FHMoKZcS)
- Beyond Scalar Reward Model: Learning Generative Judge from Preference Data
  ICLR 2025 [**Paper**](https://arxiv.org/abs/2410.03742)
- Large language models are zero-shot rankers for recommender systems
  
- Position bias in multiple-choice questions
- JurEE not Judges: safeguarding llm interactions with small, specialised Encoder Ensembles
- Split and merge: Aligning position biases in large language model based evaluators
- Large language models are not fair evaluators
- Debating with more persuasive llms leads to more truthful answers
- Position bias estimation for unbiased learning to rank in personal search
- Prd: Peer rank and discussion improve large language model based evaluations
- Humans or llms as the judge? a study on judgement biases
- Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on Zero-shot LLM Assessment
- JUDING THE JUDGES: ASYSTEMATIC INVESTIGATION OF POSITION BIAS IN PAIRWISE COMPARATIVE AS
- Generative judge for evaluating alignment
- Judging llm-as-a-judge with mt-bench and chatbot arena
- Justice or prejudice? quantifying biases in llm-as-a-judge
  NeurIPS 2024 Workshop SafeGenAi 2024 [**Paper**](https://openreview.net/pdf?id=wtscPS2zJH)
### 5.1.2 Social-Related Biases
- Benchmarking cognitive biases in large language models as evaluators
- Justice or prejudice? quantifying biases in llm-as-a-judge
- Humans or llms as the judge? a study on judgement biases
### 5.1.3 Content-Related Biases
- Calibrate before use: Improving few-shot performance of language models
- Mitigating label biases for in-context learning
- Prototypical calibration for few-shot learning of language models
- Justice or prejudice? quantifying biases in llm-as-a-judge
- Are Large Language Models Rational Investors?
- Bias patterns in the application of LLMs for clinical decision support: A comprehensive study
- Batch calibration: Rethinking calibration for in-context learning and prompt engineering
### 5.1.4 Cognitive-Related Biases
- Justice or prejudice? quantifying biases in llm-as-a-judge
- Large language models can be easily distracted by irrelevant context
- Reference-Guided Verdict: LLMs-as-Judges in Automatic Evaluation of Free-Form Text
- Prd: Peer rank and discussion improve large language model based evaluations
- Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement
- Pride and prejudice: LLM amplifies self-bias in self-refinement
- Humans or llms as the judge? a study on judgement biases
- Evaluations of self and others: Self-enhancement biases in social judgments
- Judging llm-as-a-judge with mt-bench and chatbot arena
- G-eval: Nlg evaluation using gpt-4 with better human alignment
- Benchmarking cognitive biases in large language models as evaluators
- Debating with more persuasive llms leads to more truthful answers
## 5.2 Adversarial Attacks
### 5.2.1 Adversarial Attacks on LLMs
- Hotflip: White-box adversarial examples for text classification
- Query-efficient and scalable black-box adversarial attacks on discrete sequential data via bayesian optimization
- Adv-bert: Bert is not robust on misspellings! generating nature adversarial samples on bert
- An LLM can Fool Itself: A Prompt-Based Adversarial Attack
- Natural backdoor attack on text data
- Ignore previous prompt: Attack techniques for language models
- Prompt packer: Deceiving llms through compositional instruction with hidden attacks
- Evaluating the susceptibility of pre-trained language models via handcrafted adversarial examples
### 5.2.2 Adversarial Attacks on LLMs-as-Judges
- Llama 2: Open foundation and fine-tuned chat models
- Cheating automatic llm benchmarks: Null models achieve high win rates
- Optimization-based Prompt Injection Attack to LLM-as-a-Judge
- Finding Blind Spots in Evaluator LLMs with Interpretable Checklists
- Judging llm-as-a-judge with mt-bench and chatbot arena
- Scaling instruction-finetuned language models
- Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on Zero-shot LLM Assessment

## 5.3 Inherent Weaknesses
### 5.3.1 Knowledge Recency
- Retrieval-augmented generation for large language models: A survey
- Retrieval-augmented generation for knowledge-intensive nlp tasks
- An empirical study of catastrophic forgetting in large language models during continual fine-tuning
- Continual learning for large language models: A survey
- Striking the balance in using LLMs for fact-checking: A narrative literature review

### 5.3.2 Hallucination
- Striking the balance in using LLMs for fact-checking: A narrative literature review

### 5.3.3 Domain-Specific Knowledge Gaps
- Knowledge solver: Teaching llms to search for domain knowledge from knowledge graphs
- Unifying large language models and knowledge graphs: A roadmap
- Retrieval-augmented generation for large language models: A survey

