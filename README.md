# Synthatic data generation with Differential Privacy
Multiprocessing pipeline with the Producer/Consumer pattern for synthetic tabular data generation with LLMs. Uses Hugging Face + LangChain to build prompts from seed data, generate rows, apply differential privacy sampling, and write results to CSV with optional JSONL logging. Supports multi-GPU execution.

⚠️ Note: This repository is provided as an example implementation only.
It is not a polished production system 
— you may need to adapt, extend, or harden it for your own use cases.

### DP Pipeline Overview

```mermaid
flowchart LR
    A[Extract n random example rows from file] --> B[Enrich and create prompt]
    B --> C[Call LLM with prompt]
    C --> D[Validate output with Pydantic]
    D -->|valid| E[Add DP noise probabilities]
    E --> G[Append result to CSV file]
    D -->|invalid| F[Log error and raw output to JSONL file]

    classDef defaultPath fill:#eaffea,stroke:#33aa33,stroke-width:2px;
    class E,G defaultPath;
```
