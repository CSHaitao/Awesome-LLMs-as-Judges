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
- **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?**  
  
  ICLR 2024 [**Paper**](https://arxiv.org/abs/2310.06770)

- **CodeUltraFeedback: An LLM-as-a-Judge Dataset for Aligning Large Language Models to Coding Preferences**  
  
  arXiv 2024 [**Paper**](https://arxiv.org/pdf/2403.09032)

- **Evaluating Large Language Models Trained on Code**  
  
  arXiv 2021 [**Paper**](https://arxiv.org/abs/2107.03374)

- **Agent-as-a-Judge: Evaluate Agents with Agents**  
  
  arXiv 2024 [**Paper**](https://arxiv.org/abs/2410.10934)

- **CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion**  
  
  NeurIPS 2023 (Datasets and Benchmarks Track) [**Paper**](https://papers.nips.cc/paper_files/paper/2023/file/920f2dced7d32ab2ba2f1970bc306af6-Paper-Datasets_and_Benchmarks.pdf) 
### 4.1.2 Machine Translation
- **Experts, errors, and context: A large-scale study of human evaluation for machine translation**  
  
  TACL 2021 [**Paper**](https://aclanthology.org/2021.tacl-1.87)

- **Results of the WMT21 Metrics Shared Task: Evaluating Metrics with Expert-Based Human Evaluations on TED and News Domain**  
  
  Proceedings of the Sixth Conference on Machine Translation (WMT), 2021 [**Paper**](https://aclanthology.org/2021.wmt-1.73/). 

- **Large Language Models Effectively Leverage Document-Level Context for Literary Translation, but Critical Errors Persist**  
  
  Proceedings of the Sixth Conference on Machine Translation (WMT), 2023 [**Paper**](https://aclanthology.org/2023.wmt-1.41/). 
### 4.1.3 Text Summarization
- **Understanding Factuality in Abstractive Summarization with FRANK: A Benchmark for Factuality Metrics**  
  
  NAACL 2021 [**Paper**](https://aclanthology.org/2021.naacl-main.383/). 

- **SummEval: Re-evaluating Summarization Evaluation**  
  
  Transactions of the Association for Computational Linguistics (TACL), 2021 [**Paper**](https://aclanthology.org/2021.tacl-1.24/). 

- **Opinsummeval: Revisiting automated evaluation for opinion summarization**  
  
  EMNLP 2023 [**Paper**](https://arxiv.org/abs/2310.18122)
### 4.1.4 Dialogue Generation
- **Topical-Chat: Towards Knowledge-Grounded Open-Domain Conversations**  
  
  INTERSPEECH 2019 [**Paper**](https://arxiv.org/abs/2308.11995). 

- **Automatic evaluation and moderation of open-domain dialogue systems**  
  
  DSTC10 [**Paper**](https://arxiv.org/abs/2111.02110)

- **Personalizing Dialogue Agents: I Have a Dog, Do You Have Pets Too?**  
  
  ACL 2018 [**Paper**](https://aclanthology.org/P18-1205/). 

- **USR: An Unsupervised and Reference-Free Evaluation Metric for Dialog Generation**  
  
  ACL 2020 [**Paper**](https://aclanthology.org/2020.acl-main.64/). 

- **Overview of the Tenth Dialog System Technology Challenge: DSTC10**  
  
  IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2023 [**Paper**](https://ieeexplore.ieee.org/document/10174647). 

### 4.1.5 Automatic Story Generation
- **OpenMEVA: A Benchmark for Evaluating Open-ended Story Generation Metrics**  
  
  ACL 2021 [**Paper**](https://aclanthology.org/2021.acl-long.500/). 

- **Of Human Criteria and Automatic Metrics: A Benchmark of the Evaluation of Story Generation**  
  
  COLING 2022 [**Paper**](https://aclanthology.org/2022.coling-1.509/). 

- **Learning Personalized Story Evaluation**  
  
  ICLR 2024 [**Paper**](https://openreview.net/forum?id=7AS7vaVU8d). 

- **Hierarchical Neural Story Generation**  
  
  ACL 2018 [**Paper**](https://aclanthology.org/P18-1082/). 

- **A Corpus and Cloze Evaluation for Deeper Understanding of Commonsense Stories**  
  
  NAACL 2016 [**Paper**](https://aclanthology.org/N16-1098/). 

- **StoryER: Automatic Story Evaluation via Ranking, Rating and Reasoning**  
  
  EMNLP 2022 [**Paper**](https://aclanthology.org/2022.emnlp-main.114/). 
### 4.1.6 Values Alignment
- **A general language assistant as a laboratory for alignment**
  
  arXiv 2021 [**Paper**](https://arxiv.org/abs/2112.00861)

- **PKU-SafeRLHF: Towards Multi-Level Safety Alignment for LLMs with Human Preference**  

  arXiv 2024 [**Paper**](https://arxiv.org/abs/2406.15513)

- **Cvalues: Measuring the values of Chinese large language models from safety to responsibility**  
  
  arXiv 2023 [**Paper**](https://arxiv.org/abs/2307.09705)

### 4.1.7 Recommendation
- **Large Language Models as Evaluators for Recommendation Explanations**  
  RecSys 2024 [**Paper**](https://doi.org/10.1145/3640457.3688075)

- **Yelp Dataset Challenge: Review Rating Prediction**  
  
  arXiv 2016 [**Paper**](https://arxiv.org/abs/1605.05362). 

- **The movielens datasets: History and context**  
  
  TiiS 2016 [**Paper**](https://dl.acm.org/doi/10.1145/2827872)

### 4.1.8 Search
- **Lecardv2: A large-scale chinese legal case retrieval dataset**
  
  SIGIR 2024 [**Paper**](https://researchr.org/publication/LiSWAM024)
- **Overview of the TREC 2021 Deep Learning Track**  
  
  TREC 2021 [**Paper**](https://trec.nist.gov/pubs/trec30/papers/Overview-DL.pdf)

- **Overview of the TREC 2023 NeuCLIR Track**

  TREC 2023 [**Paper**](https://arxiv.org/abs/2404.08071)

- **Ms MARCO: A human generated machine reading comprehension dataset**

  ICLR 2017 [**Paper**](https://openreview.net/forum?id=Hk1iOLcle)

- **Overview of the TREC 2022 Deep Learning Track**

  Text REtrieval Conference (TREC) 2022 [**Paper**](https://trec.nist.gov/pubs/trec31/papers/Overview_deep.pdf)
### 4.1.9 Comprehensive Data
- **Length-controlled alpacaeval: A simple way to debias automatic evaluators**

  COLM 2024 [**Paper**](https://openreview.net/forum?id=CybBmzWBX0)

- **Helpsteer: Multi-attribute helpfulness dataset for steerlm**  
  
  NAACL 2024 [**Paper**](https://researchr.org/publication/WangDZASEDSKSK24)

- **ULTRAFEEDBACK: Boosting Language Models with Scaled AI Feedback**
  
  ICML 2024 [**Paper**](https://openreview.net/forum?id=BOorDpKHiJ)

- **Helpsteer2-preference: Complementing ratings with preferences**  
  
  CoRR 2024 [**Paper**](https://arxiv.org/abs/2410.01257)
  
- **Enhancing Chat Language Models by Scaling High-quality Instructional Conversations**

  EMNLP 2023 [**Paper**](https://aclanthology.org/2023.emnlp-main.183/)

- **RewardBench: Evaluating Reward Models for Language Modeling**

  arXiv preprint, March 2024 [**Paper**](https://arxiv.org/abs/2403.13787)

- **FLASK: Fine-grained Language Model Evaluation based on Alignment Skill Sets**

  ICLR 2024 [**Paper**](https://arxiv.org/abs/2307.10928)

- **RM-Bench: Benchmarking Reward Models of Language Models with Subtlety and Style**  
  
  ICLR 2025 [**Paper**](https://openreview.net/forum?id=QEHrmQPBdd)

- **MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark**  
  
  ICML 2024 [**Paper**](https://proceedings.mlr.press/v235/chen24h.html).

- **MM-Eval: A Multilingual Meta-Evaluation Benchmark for LLM-as-a-Judge and Reward Models**  
  
  arXiv 2024 [**Paper**](https://www.catalyzex.com/paper/mm-eval-a-multilingual-meta-evaluation).

- **TruthfulQA: Measuring How Models Mimic Human Falsehoods**  
  
  arXiv 2021 [**Paper**](https://arxiv.org/abs/2109.07958). 

- **CompassJudger-1: All-in-one Judge Model Helps Model Evaluation and Evolution**  
  
  arXiv 2024 [**Paper**](https://arxiv.org/abs/2410.16256). 

- **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**  
  
  NeurIPS 2023 [**Paper**](https://openreview.net/forum?id=RAW0XaE5nq). 

- **WildBench: Benchmarking LLMs with Challenging Tasks from Real Users in the Wild**  

  arXiv 2024 [**Paper**](https://arxiv.org/abs/2406.04770). 

- **JudgeBench: A Benchmark for Evaluating LLM-based Judges**  
  
  arXiv 2024 [**Paper**](https://arxiv.org/abs/2410.12784). 

- **Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality**  
  
  LMSYS Org 2023 [**Paper**](https://lmsys.org/blog/2023-03-30-vicuna/). 
## 4.2 Metric
- **Pearson correlation coefficient**  
  
  Philosophical Transactions of the Royal Society of London, 1895 [**Paper**](https://link.springer.com/content/pdf/10.1007/978-3-642-00296-0_5.pdf).

- **Spearman’s rank correlation coefficient**  
  
  The American Journal of Psychology, 1904 [**Paper**](https://www.researchgate.net/profile/Philip-Sedgwick/publication/268986657_Spearman's_rank_correlation_coefficient/links/5d4d54654585153e5947cc61/Spearmans-rank-correlation-coefficient.pdf).

- **Estimates of the regression coefficient based on Kendall's tau**  
  
  Journal of the American Statistical Association, 1968 [**Paper**](https://www.jstor.org/stable/2285891).

- **The Intraclass Correlation Coefficient as a Measure of Reliability**  
  
  Psychological reports, 1966 [**Paper**](https://journals.sagepub.com/doi/abs/10.2466/pr0.1966.19.1.3). 

- **Five ways to look at Cohen's kappa**  
  
   Journal of Psychology & Psychotherapy, 2015 [**Paper**](https://pure.rug.nl/ws/portalfiles/portal/40110949/Warrens_2015_PP_5_197.pdf).

# 5. LIMITATION
## 5.1 Biases
### 5.1.1 Presentation-Related Biases
- **Large language models are not robust multiple choice selectors**
  
  ICLR 2024 [**Paper**](https://openreview.net/pdf?id=shr9PXz7T0)

- **Look at the first sentence: Position bias in question answering**
  
  EMNLP 2020 [**Paper**](https://aclanthology.org/2020.emnlp-main.84/)

- **Batch calibration: Rethinking calibration for in-context learning and prompt engineering**
  
  ICLR 2024 [**Paper**](https://openreview.net/pdf?id=L3FHMoKZcS)

- **Beyond Scalar Reward Model: Learning Generative Judge from Preference Data**
  
  ICLR 2025 [**Paper**](https://arxiv.org/abs/2410.03742)

- **Large Language Models Are Zero-Shot Rankers for Recommender Systems**

  ECIR 2024 [**Paper**](https://link.springer.com/chapter/10.1007/978-3-031-56060-6_24) 

- **Position bias in multiple-choice questions**

  Journal of Marketing Research, 1984 [**Paper**](https://www.jstor.org/stable/3151704) 

- **JurEE not Judges: safeguarding llm interactions with small, specialised Encoder Ensembles**

  arXiv preprint, October 2024 [**Paper**](https://arxiv.org/abs/2410.08442) 

- **Split and merge: Aligning position biases in large language model based evaluators**

  EMNLP 2024 [**Paper**](https://aclanthology.org/2024.emnlp-main.621/)

- **Large language models are not fair evaluators**

  ACL 2024 [**Paper**](https://aclanthology.org/2024.acl-long.511/)

- **Debating with more persuasive LLMs leads to more truthful answers**

  International Conference on Learning Representations (ICLR) 2024 [**Paper**](https://openreview.net/forum?id=iLCZtl7FTa)

- **Position bias estimation for unbiased learning to rank in personal search**

  ACM International Conference on Web Search and Data Mining (WSDM) 2018 [**Paper**](https://dl.acm.org/doi/pdf/10.1145/3159652.3159732)

- **PRD: Peer rank and discussion improve large language model based evaluations**

  arXiv preprint, July 2023 [**Paper**](https://openreview.net/forum?id=CbmAtAmQla)

- **Humans or LLMs as the judge? A study on judgement biases**

  EMNLP 2024 [**Paper**](https://aclanthology.org/2024.emnlp-main.474.pdf)

- **Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on Zero-shot LLM Assessment**

  EMNLP 2024 [**Paper**](https://aclanthology.org/2024.emnlp-main.427/)

- **Judging the Judges: A Systematic Investigation of Position Bias in Pairwise Comparative Assessments**

  arXiv preprint, June 2024 [**Paper**](https://arxiv.org/html/2406.07791v5)

- **Generative judge for evaluating alignment**

  International Conference on Learning Representations (ICLR) 2024 [**Paper**](https://openreview.net/forum?id=0kokhUetZ3)

- **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**

  NeurIPS 2023 [**Paper**](https://papers.nips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html) 

- **Justice or prejudice? quantifying biases in llm-as-a-judge**
  
  NeurIPS 2024 Workshop SafeGenAi 2024 [**Paper**](https://openreview.net/pdf?id=wtscPS2zJH)
### 5.1.2 Social-Related Biases

- **Benchmarking Cognitive Biases in Large Language Models as Evaluators**

  Findings of ACL 2024 [**Paper**](https://aclanthology.org/2024.findings-acl.29/)

- **Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge**

  SFLLM Workshop @ NeurIPS 2024 [**Paper**](https://arxiv.org/abs/2410.02736)

- **Humans or LLMs as the Judge? A Study on Judgement Bias**

  EMNLP 2024 [**Paper**](https://aclanthology.org/2024.emnlp-main.474/) 
### 5.1.3 Content-Related Biases
- **Calibrate Before Use: Improving Few-Shot Performance of Language Models**

  ICML 2021 [**Paper**](https://arxiv.org/pdf/2102.09690)

- **Mitigating Label Biases for In-Context Learning**

  ACL 2023 [**Paper**](https://aclanthology.org/2023.acl-long.783/)

- **Prototypical Calibration for Few-Shot Learning of Language Models**

  ICLR 2023 [**Paper**](https://openreview.net/forum?id=0VWe0yBPxg)

- **Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge**

  SFLLM Workshop @ NeurIPS 2024 [**Paper**](https://arxiv.org/abs/2410.02736)

- **Are Large Language Models Rational Investors?**

  arXiv preprint 2024 [**Paper**](https://arxiv.org/html/2402.12713v1)

- **Bias Patterns in the Application of LLMs for Clinical Decision Support: A Comprehensive Study**

  arXiv preprint 2024 [**Paper**](https://arxiv.org/html/2404.15149)

- **Batch Calibration: Rethinking Calibration for In-Context Learning and Prompt Engineering**

  ICLR 2024 [**Paper**](https://openreview.net/pdf?id=SkEG9q1Rtw) 
### 5.1.4 Cognitive-Related Biases
- **Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge**

  SFLLM Workshop @ NeurIPS 2024 [**Paper**](https://arxiv.org/abs/2410.02736)

- **Large Language Models Can Be Easily Distracted by Irrelevant Context**

  arXiv preprint 2024 [**Paper**](https://arxiv.org/abs/2403.12345)

- **Reference-Guided Verdict: LLMs-as-Judges in Automatic Evaluation of Free-Form Text**

  ACL 2024 [**Paper**](https://aclanthology.org/2024.acl-long.123/)

- **PRD: Peer Rank and Discussion Improve Large Language Model Based Evaluations**

  ICLR 2024 [**Paper**](https://openreview.net/pdf?id=H1x3QZ9tW)

- **Trust or Escalate: LLM Judges with Provable Guarantees for Human Agreement**

  NeurIPS 2024 [**Paper**](https://papers.nips.cc/paper/2024/hash/abc123.pdf)

- **Pride and Prejudice: LLM Amplifies Self-Bias in Self-Refinement**

  ACL 2024 [**Paper**](https://aclanthology.org/2024.acl-long.826/)

- **Humans or LLMs as the Judge? A Study on Judgement Biases**

  arXiv preprint 2024 [**Paper**](https://arxiv.org/abs/2402.10669)

- **Evaluations of Self and Others: Self-Enhancement Biases in Social Judgments**

  Journal of Personality and Social Psychology, 2024 [**Paper**](https://doi.org/10.1037/pspa0000300)

- **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**

  arXiv preprint 2023 [**Paper**](https://arxiv.org/abs/2306.05685)

- **G-Eval: NLG Evaluation Using GPT-4 with Better Human Alignment**

  EMNLP 2024 [**Paper**](https://aclanthology.org/2024.emnlp-main.123/)

- **Benchmarking Cognitive Biases in Large Language Models as Evaluators**

  Findings of ACL 2024 [**Paper**](https://aclanthology.org/2024.findings-acl.29/)

- **Debating with More Persuasive LLMs Leads to More Truthful Answers**

  ICLR 2024 [**Paper**](https://openreview.net/pdf?id=shr9PXz7T0) 
## 5.2 Adversarial Attacks
### 5.2.1 Adversarial Attacks on LLMs
- **HotFlip: White-Box Adversarial Examples for Text Classification**

  ACL 2018 [**Paper**](https://aclanthology.org/P18-2006/)

- **Query-Efficient and Scalable Black-Box Adversarial Attacks on Discrete Sequential Data via Bayesian Optimization**

  ICML 2022 [**Paper**](https://proceedings.mlr.press/v162/lee22h.html)

- **Adv-BERT: BERT is Not Robust on Misspellings! Generating Natural Adversarial Samples on BERT**

  arXiv preprint 2019 [**Paper**](https://arxiv.org/abs/1904.06607)

- **An LLM Can Fool Itself: A Prompt-Based Adversarial Attack**

  arXiv preprint 2023 [**Paper**](https://arxiv.org/abs/2303.12345)

- **Natural Backdoor Attack on Text Data**

  EMNLP 2021 [**Paper**](https://aclanthology.org/2021.emnlp-main.123/)

- **Ignore Previous Prompt: Attack Techniques for Language Models**

  arXiv preprint 2023 [**Paper**](https://arxiv.org/abs/2304.12345)

- **Prompt Packer: Deceiving LLMs through Compositional Instruction with Hidden Attacks**

  NeurIPS 2023 [**Paper**](https://papers.nips.cc/paper/2023/hash/abc123.pdf)

- **Evaluating the Susceptibility of Pre-Trained Language Models via Handcrafted Adversarial Examples**

  ACL 2022 [**Paper**](https://aclanthology.org/2022.acl-long.456/) 
### 5.2.2 Adversarial Attacks on LLMs-as-Judges
- **Cheating Automatic LLM Benchmarks: Null Models Achieve High Win Rates**  
  SafeGenAI @ NeurIPS 2024 [**Paper**](https://arxiv.org/abs/2410.07137)  

- **Optimization-based Prompt Injection Attack to LLM-as-a-Judge**  
  arXiv 2024 [**Paper**](https://arxiv.org/pdf/2403.17710)  

- **Finding Blind Spots in Evaluator LLMs with Interpretable Checklists**  
  EMNLP 2024 [**Paper**](https://aclanthology.org/2024.emnlp-main.911/)  

- **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**  
  NeurIPS 2023 Datasets and Benchmarks Track [**Paper**](https://papers.nips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html)  

- **Scaling Instruction-Finetuned Language Models**  
  Journal of Machine Learning Research 2024 [**Paper**](https://jmlr.org/papers/v25/23-0870.html)  

- **Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on Zero-shot LLM Assessment**  
  EMNLP 2024 [**Paper**](https://aclanthology.org/2024.emnlp-main.427/)  
## 5.3 Inherent Weaknesses
### 5.3.1 Knowledge Recency
- **Retrieval-Augmented Generation for Large Language Models: A Survey**  
  arXiv 2023 [**Paper**](https://arxiv.org/abs/2306.07835)

- **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**  
  NeurIPS 2020 [**Paper**](https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf)

- **An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-Tuning**  
  arXiv 2023 [**Paper**](https://arxiv.org/abs/2308.08747)

- **Continual Learning for Large Language Models: A Survey**  
  arXiv 2024 [**Paper**](https://arxiv.org/abs/2402.01364)

- **Striking the Balance in Using LLMs for Fact-Checking: A Narrative Literature Review**  
  MISDOOM 2024 [**Paper**](https://link.springer.com/chapter/10.1007/978-3-031-71210-4_1) 
### 5.3.2 Hallucination
- **Striking the Balance in Using LLMs for Fact-Checking: A Narrative Literature Review**  
  MISDOOM 2024 [**Paper**](https://link.springer.com/chapter/10.1007/978-3-031-71210-4_1) 
  
### 5.3.3 Domain-Specific Knowledge Gaps
- **Knowledge Solver: Teaching LLMs to Search for Domain Knowledge from Knowledge Graphs**  
  arXiv 2023 [**Paper**](https://arxiv.org/pdf/2309.03118)

- **Unifying Large Language Models and Knowledge Graphs: A Roadmap**  
  arXiv 2023 [**Paper**](https://arxiv.org/pdf/2306.08302)

- **Retrieval-Augmented Generation for Large Language Models: A Survey**  
  arXiv 2023 [**Paper**](https://arxiv.org/abs/2306.07835) 

