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

## Overview of Awesome-llm-as-judges:
![Overview](./img/overview.png)
![limit](./img/limit.png)

# 1. Functionality

## 1.1 Performance Evaluation
### 1.1.1 Responses Evaluation
- **Llm-eval: Unified multi-dimensional automatic evaluation for open-domain conversations with large language models**. arXiv preprint arXiv:2305.13711 (2023). [[Paper](https://arxiv.org/abs/2305.13711)]
- **Automated Genre-Aware Article Scoring and Feedback Using Large Language Models**. arXiv preprint arXiv:2410.14165 (2024). [[Paper](https://arxiv.org/abs/2410.14165)]
- **Is LLM a Reliable Reviewer? A Comprehensive Evaluation of LLM on Automatic Paper Reviewing Tasks**. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024). 9340–9351.[[Paper](https://aclanthology.org/2024.lrec-main.816/)]
- **Ares: An automated evaluation framework for retrieval-augmented generation systems**. arXiv preprint arXiv:2311.09476 (2023). [[Paper](https://arxiv.org/abs/2311.09476)]
- **Self-rag: Learning to retrieve, generate, and critique through self-reflection**. arXiv preprint arXiv:2310.11511 (2023). [[Paper](https://arxiv.org/abs/2310.11511)]
- **RecExplainer: Aligning Large Language Models for Explaining Recommendation Models**. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 1530–1541.[[Paper](https://dl.acm.org/doi/10.1145/3637528.3671802)]
### 1.1.2 Model Evaluation
- **Judging llm-as-a-judge with mt-bench and chatbot arena**. Advances in Neural Information Processing Systems 36 (2023), 46595–46623.[[Paper](https://arxiv.org/abs/2306.05685)]
- **Auto Arena of LLMs: Automating LLM Evaluations with Agent Peer-battles and Committee Discussions**. arXiv preprint arXiv:2405.20267 (2024). [[Paper](https://arxiv.org/abs/2405.20267)]
- **VideoAutoArena: An Automated Arena for Evaluating Large Multimodal Models in Video Analysis through User Simulation**. arXiv preprint arXiv:2411.13281 (2024). [[Paper](https://arxiv.org/abs/2411.13281)]
- **Benchmarking foundation models with language-model-as-an-examiner**. Advances in Neural Information Processing Systems 36 (2024).[[Paper](https://arxiv.org/abs/2306.04181)]
- **Kieval: A knowledge-grounded interactive evaluation framework for large language models**. arXiv preprint arXiv:2402.15043 (2024). [[Paper](https://arxiv.org/abs/2402.15043)]
## 1.2 Model Enhancement
### 1.2.1 Reward Modeling During Training
- **Self-rewarding language models**. arXiv preprint arXiv:2401.10020 (2024). [[Paper](https://arxiv.org/abs/2401.10020)]
- **Direct language model alignment from online ai feedback**. arXiv preprint arXiv:2402.04792 (2024). [[Paper](https://arxiv.org/abs/2402.04792)]
- **Rlaif: Scaling reinforcement learning from human feedback with ai feedback**. arXiv preprint arXiv:2309.00267 (2023).[[Paper](https://arxiv.org/abs/2309.00267)]
- **Enhancing Reinforcement Learning with Dense Rewards from Language Model Critic**. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 9119–9138.[[Paper](https://aclanthology.org/2024.emnlp-main.515/)]
- **Cream: Consistency regularized self-rewarding language models**. arXiv preprint arXiv:2410.12735 (2024). [[Paper](https://arxiv.org/abs/2410.12735)]
- **The perfect blend: Redefining RLHF with mixture of judges**. arXiv preprint arXiv:2409.20370 (2024). [[Paper](https://arxiv.org/abs/2409.20370)]

### 1.2.2 Acting as Verifier During Inference
- **Regularized Best-of-N Sampling to Mitigate Reward Hacking for Language Model Alignment**. arXiv preprint arXiv:2404.01054 (2024). [[Paper](https://arxiv.org/abs/2404.01054)]
- **Fast Best-of-N Decoding via Speculative Rejection**. arXiv preprint arXiv:2410.20290 (2024). [[Paper](https://arxiv.org/abs/2410.20290)]
- **Tree of thoughts: Deliberate problem solving with large language models**. Advances in Neural Information Processing Systems 36 (2024).[[Paper](https://arxiv.org/abs/2305.10601)]
- **Graph of thoughts: Solving elaborate problems with large language models**. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 17682–17690.[[Paper](https://arxiv.org/abs/2308.09687)]
- **Let’s verify step by step**. arXiv preprint arXiv:2305.20050 (2023). [[Paper](https://arxiv.org/abs/2305.20050)]
- **Self-evaluation guided beam search for reasoning**. Advances in Neural Information Processing Systems 36 (2024).[[Paper](https://arxiv.org/abs/2305.00633)]
- **Rationale-Aware Answer Verification by Pairwise Self-Evaluation**. arXiv preprint arXiv:2410.04838 (2024). [[Paper](https://arxiv.org/abs/2410.04838)]
- **Creative Beam Search: LLM-as-a-Judge for Improving Response Generation**. ICCC (2024).[[Paper](https://arxiv.org/abs/2405.00099)]


### 1.2.3 Feedback for Refinement
- **Self-refine: Iterative refinement with self-feedback**. Advances in Neural Information Processing Systems 36 (2024).[[Paper](https://arxiv.org/abs/2303.17651)]
- **Teaching large language models to self-debug**. arXiv preprint arXiv:2304.05128 (2023). [[Paper](https://arxiv.org/abs/2304.05128)]
- **Refiner: Reasoning feedback on intermediate representations**. arXiv preprint arXiv:2304.01904 (2023). [[Paper](https://arxiv.org/abs/2304.01904)]
- **Towards reasoning in large language models via multi-agent peer review collaboration**. arXiv preprint arXiv:2311.08152 (2023). [[Paper](https://arxiv.org/abs/2311.08152)]
- **Large language models cannot self-correct reasoning yet**. arXiv preprint arXiv:2310.01798 (2023). [[Paper](https://arxiv.org/abs/2310.01798)]
- **LLMs cannot find reasoning errors, but can correct them!**. arXiv preprint arXiv:2311.08516 (2023). [[Paper](https://arxiv.org/abs/2311.08516)]
- **Can large language models really improve by self-critiquing their own plans?**. arXiv preprint arXiv:2310.08118 (2023). [[Paper](https://arxiv.org/abs/2310.08118)]

## 1.3 Data Collection
### 1.3.1 Data Annotation
- **If in a Crowdsourced Data Annotation Pipeline, a GPT-4**. In Proceedings of the CHI Conference on Human Factors in Computing Systems. 1–25.[[Paper](https://arxiv.org/abs/2402.16795)]
- **ChatGPT outperforms crowd workers for text-annotation tasks**. Proceedings of the National Academy of Sciences 120, 30 (2023), e2305016120.[[Paper](https://arxiv.org/abs/2303.15056)]
- **ChatGPT-4 outperforms experts and crowd workers in annotating political Twitter messages with zero-shot learning**. arXiv preprint arXiv:2304.06588 (2023). [[Paper](https://arxiv.org/abs/2304.06588)]
- **Fullanno: A data engine for enhancing image comprehension of MLLMs**. arXiv preprint arXiv:2409.13540 (2024). [[Paper](https://arxiv.org/abs/2409.13540)]
- **Can large language models aid in annotating speech emotional data? Uncovering new frontiers**. arXiv preprint arXiv:2307.06090 (2023). [[Paper](https://arxiv.org/abs/2307.06090)]
- **Annollm: Making large language models to be better crowdsourced annotators**. arXiv preprint arXiv:2303.16854 (2023). [[Paper](https://arxiv.org/abs/2303.16854)]
- **LLMAAA: Making large language models as active annotators**. arXiv preprint arXiv:2310.19596 (2023). [[Paper](https://arxiv.org/abs/2310.19596)]

### 1.3.2 Data Synthesize
- **Selfee: Iterative self-revising LLM empowered by self-feedback generation**. Blog post (2023).[[Blog](https://lklab.kaist.ac.kr/SelFee/)]
- **Self-Boosting Large Language Models with Synthetic Preference Data**. arXiv preprint arXiv:2410.06961 (2024). [[Paper](https://arxiv.org/abs/2410.06961)]
- **The fellowship of the LLMs: Multi-agent workflows for synthetic preference optimization dataset generation**. arXiv preprint arXiv:2408.08688 (2024). [[Paper](https://arxiv.org/abs/2408.08688)]
- **Self-consistency improves chain of thought reasoning in language models**. arXiv preprint arXiv:2203.11171 (2022). [[Paper](https://arxiv.org/abs/2203.11171)]
- **WizardLM: Empowering large language models to follow complex instructions**. arXiv preprint arXiv:2304.12244 (2023). [[Paper](https://arxiv.org/abs/2304.12244)]
- **Automatic Instruction Evolving for Large Language Models**. arXiv preprint arXiv:2406.00770 (2024). [[Paper](https://arxiv.org/abs/2406.00770)]
- **STaR: Self-taught reasoner bootstrapping reasoning with reasoning**. In Proc. the 36th International Conference on Neural Information Processing Systems, Vol. 1126.[[Paper](https://arxiv.org/abs/2203.14465)]
- **Beyond human data: Scaling self-training for problem-solving with language models**. arXiv preprint arXiv:2312.06585 (2023). [[Paper](https://arxiv.org/abs/2312.06585)]


# 2. METHODOLOGY
## 2.1 Single-LLM System
### 2.1.1 Prompt-based
#### 2.1.1.1 In-Context Learning
- A survey on in-context learning
arXiv 2022 [Paper](https://arxiv.org/abs/2301.00234)
- Gptscore: Evaluate as you desire
arXiv 2023 [Paper](https://arxiv.org/abs/2302.04166)
- Llm-eval: Unified multi-dimensional automatic evaluation for open-domain conversations with large language models
NLP4ConvAI 2023 [Paper](https://arxiv.org/abs/2305.13711)
- TALEC: Teach Your LLM to Evaluate in Specific Domain with In-house Criteria by Criteria Division and Zero-shot Plus Few-shot
arXiv 2024 [Paper](https://arxiv.org/abs/2407.10999)
- Multi-dimensional evaluation of text summarization with in-context learning
ACL Findings '23 [Paper](https://arxiv.org/abs/2306.01200)
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
- **DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset**. In Proceedings of The 8th International Joint Conference on Natural Language Processing (IJCNLP 2017).[[Poster](https://arxiv.org/abs/1710.03957)]
- **Don’t give me the details, just the summary! Topic-aware convolutional neural networks for extreme summarization**. arXiv preprint arXiv:1808.08745 (2018). [[Paper](https://arxiv.org/abs/1808.08745)]
- **Improving LLM-based machine translation with systematic self-correction**. arXiv preprint arXiv:2402.16379 (2024). [[Paper](https://arxiv.org/abs/2402.16379)]
- **Fusion-Eval: Integrating Assistant Evaluators with LLMs**. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track. 225–238.[[Poster](https://arxiv.org/abs/2311.09204)]


## 3.2 Multimodal
- **Llava-critic: Learning to evaluate multimodal models**. arXiv preprint arXiv:2410.02712 (2024). [[Paper](https://arxiv.org/abs/2410.02712)]
- **Mllm-as-a-judge: Assessing multimodal llm-as-a-judge with vision-language benchmark**. arXiv preprint arXiv:2402.04788 (2024). [[Paper](https://arxiv.org/abs/2402.04788)]
- **Can large language models aid in annotating speech emotional data? uncovering new frontiers**. arXiv preprint arXiv:2307.06090 (2023). [[Paper](https://arxiv.org/abs/2307.06090)]
- **Efficient Self-Improvement in Multimodal Large Language Models: A Model-Level Judge-Free Approach**. arXiv preprint arXiv:2411.17760 (2024). [[Paper](https://arxiv.org/abs/2411.17760)]
- **Calibrated self-rewarding vision language models**. arXiv preprint arXiv:2405.14622 (2024). [[Paper](https://arxiv.org/abs/2405.14622)]
- **Automated evaluation of large vision-language models on self-driving corner cases**. arXiv preprint arXiv:2404.10595 (2024). [[Paper](https://arxiv.org/abs/2404.10595)]

## 3.3 Medical
- **DOCLENS: Multi-aspect fine-grained evaluation for medical text generation**. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics. (2024).[[Paper](https://arxiv.org/abs/2311.09581)]
- **Comparing Two Model Designs for Clinical Note Generation; Is an LLM a Useful Evaluator of Consistency?**. arXiv preprint arXiv:2404.06503 (2024). [[Paper](https://arxiv.org/abs/2404.06503)]
- **Towards Leveraging Large Language Models for Automated Medical Q&A Evaluation**. arXiv preprint arXiv:2409.01941 (2024). [[Paper](https://arxiv.org/abs/2409.01941)]
- **Automatic evaluation for mental health counseling using LLMs**. arXiv preprint arXiv:2402.11958 (2024). [[Paper](https://arxiv.org/abs/2402.11958)]
- **Improving medical reasoning through retrieval and self-reflection with retrieval-augmented large language models**. Bioinformatics 40, Supplement_1 (2024), i119–i129. [[Paper](https://arxiv.org/abs/2401.15269)]
- **DALK: Dynamic Co-Augmentation of LLMs and KG to answer Alzheimer's Disease Questions with Scientific Literature**. arXiv preprint arXiv:2405.04819 (2024). [[Paper](https://arxiv.org/abs/2405.04819)]
  
## 3.4 Legal
- **Disc-lawllm: Fine-tuning large language models for intelligent legal services**. arXiv preprint arXiv:2309.11325 (2023). [[Paper](https://arxiv.org/abs/2309.11325)]
- **Retrieval-based Evaluation for LLMs: A Case Study in Korean Legal QA**. In Proceedings of the Natural Legal Language Processing Workshop 2023. 132–137. [[Paper](https://arxiv.org/abs/2309.11325)]
- **Constructing domain-specific evaluation sets for llm-as-a-judge**. arXiv preprint arXiv:2408.08808 (2024). [[Paper](https://arxiv.org/abs/2408.08808)]
- **Leveraging Large Language Models for Relevance Judgments in Legal Case Retrieval**. arXiv preprint arXiv:2403.18405 (2024). [[Paper](https://arxiv.org/abs/2403.18405)]


## 3.5 Financial
- **Pixiu: A large language model, instruction data and evaluation benchmark for finance**. arXiv preprint arXiv:2306.05443 (2023). [[Paper](https://arxiv.org/abs/2306.05443)]
- **GPT classifications, with application to credit lending**. Machine Learning with Applications 16 (2024), 100534. [[Paper](https://doi.org/10.1016/j.mlwa.2024.100534)]
- **KRX Bench: Automating Financial Benchmark Creation via Large Language Models**. In Proceedings of the Joint Workshop of the 7th Financial Technology and Natural Language Processing, the 5th Knowledge Discovery from Unstructured Data in Financial Services, and the 4th Workshop on Economics and Natural Language Processing@ LREC-COLING 2024. 10–20.[[Paper](https://aclanthology.org/2024.finnlp-1.2/)]

## 3.6 Education
- **Large Language Model as an Assignment Evaluator: Insights, Feedback, and Challenges in a 1000+ Student Course**. arXiv preprint arXiv:2407.05216 (2024). [[Paper](https://arxiv.org/abs/2407.05216)]
- **Automated Genre-Aware Article Scoring and Feedback Using Large Language Models**. arXiv preprint arXiv:2410.14165 (2024). [[Paper](https://arxiv.org/abs/2410.14165)]
- **Automated Essay Scoring and Revising Based on Open-Source Large Language Models**. IEEE Transactions on Learning Technologies (2024). [[Paper](https://ieeexplore.ieee.org/document/10520824)] 
- **Is LLM a Reliable Reviewer? A Comprehensive Evaluation of LLM on Automatic Paper Reviewing Tasks**. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024). 9340–9351. [[Paper](https://aclanthology.org/2024.lrec-main.816/)]
- **Evaluating Mathematical Reasoning Beyond Accuracy**. arXiv preprint arXiv:2404.05692 (2024). [[Paper](https://arxiv.org/abs/2404.05692)] 
- **Debatrix: Multi-dimensional Debate Judge with Iterative Chronological Analysis Based on LLM**. arXiv preprint arXiv:2403.08010 (2024). [[Paper](https://arxiv.org/abs/2403.08010)] 


## 3.7 Information Retrieval
- **LLMJudge: LLMs for Relevance Judgments**. arXiv preprint arXiv:2408.08896 (2024). [[Paper](https://arxiv.org/abs/2408.08896)]
- **JudgeRank: Leveraging Large Language Models for Reasoning-Intensive Reranking**. arXiv preprint arXiv:2411.00142 (2024). [[Paper](https://arxiv.org/abs/2411.00142)]
- **Large language models as evaluators for recommendation explanations**. In Proceedings of the 18th ACM Conference on Recommender Systems. 33–42 (2024).
- **Ares: An automated evaluation framework for retrieval-augmented generation systems**. arXiv preprint arXiv:2311.09476 (2023). [[Paper](https://arxiv.org/abs/2311.09476)]

## 3.8 Others
- **AIME: AI System Optimization via Multiple LLM Evaluators**. arXiv preprint arXiv:2410.03131 (2024). [[Paper](https://arxiv.org/abs/2410.03131)]
- **CodeUltraFeedback: An LLM-as-a-Judge Dataset for Aligning Large Language Models to Coding Preferences**. arXiv preprint arXiv:2403.09032 (2024). [[Paper](https://arxiv.org/abs/2403.09032)]
- **LLMs as Evaluators: A Novel Approach to Evaluate Bug Report Summarization**. arXiv preprint arXiv:2409.00630 (2024). [[Paper](https://arxiv.org/abs/2409.00630)]
- **Using Large Language Models to Evaluate Biomedical Query-Focused Summarisation**. In Proceedings of the 23rd Workshop on Biomedical Natural Language Processing. 236–242 (2024).[[Paper](https://aclanthology.org/2024.bionlp-1.18/)]
- **AI can help humans find common ground in democratic deliberation**. Science 386, 6719 (2024), eadq2852.[[Paper](https://www.science.org/doi/10.1126/science.adq2852)]
- **Sotopia: Interactive evaluation for social intelligence in language agents**. arXiv preprint arXiv:2310.11667 (2023). [[Paper](https://arxiv.org/abs/2310.11667)]



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


- **The fellowship of the LLMs: Multi-agent workflows for synthetic preference optimization dataset generation**. *Samee Arif, Sualeha Farid, Abdul Hameed Azeemi, Awais Athar, Agha Ali Raza* arXiv preprint arXiv:2408.08688 (2024). [[Paper](https://arxiv.org/abs/2408.08688)]
