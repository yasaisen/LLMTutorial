# LLMTutorial

A collection of end-to-end notebooks that illustrate the core building blocks behind modern large language model (LLM) applications. Each demo focuses on a different capability—from basic chat inference, through retrieval-augmented generation (RAG) and tool use, to parameter-efficient fine-tuning with LoRA. Use these notebooks as reference implementations when prototyping your own LLM-powered systems.

## Table of Contents
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Notebook Walkthroughs](#notebook-walkthroughs)
  - [1. LLMInference\_demo.ipynb](#1-llminference_demoipynb)
  - [2. funcCalling\_demo.ipynb](#2-funccalling_demoipynb)
  - [3. RAG\_demo.ipynb](#3-rag_demoipynb)
  - [4. LoRAFinetune\_demo.ipynb](#4-lorafinetune_demoipynb)
- [Working With Your Own Data](#working-with-your-own-data)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Repository Structure

```
LLMTutorial/
├── LICENSE                    # MIT License for the project
├── README.md                  # You are here
├── LLMInference_demo.ipynb    # Basic chat inference with Gemma 3
├── funcCalling_demo.ipynb     # Function-calling agent that can use tools
├── RAG_demo.ipynb             # Retrieval-augmented generation pipeline
└── LoRAFinetune_demo.ipynb    # Parameter-efficient fine-tuning with LoRA
```

All notebooks share a set of helper utilities (`create_model`, `lm_template`, and `generate`) so you can move smoothly between inference, tool use, retrieval, and fine-tuning scenarios.

## Prerequisites

- **Python**: 3.9 or newer is recommended.
- **Hardware**: A CUDA-capable GPU with at least 16 GB of VRAM is strongly recommended for the Gemma 3 models used across the notebooks. CPU-only execution is possible for smaller workloads but will be significantly slower.
- **Hugging Face account**: The notebooks load `gemma-3-4b-it` and `google/gemma-3-1b-pt`. Make sure your account has access to these checkpoints and that you have accepted their licenses.
- **Storage**: Fine-tuning checkpoints can consume several gigabytes. Ensure you have free disk space before running the LoRA demo.

## Environment Setup

1. **Clone the repository** (if you have not already):
   ```bash
   git clone https://github.com/<your-account>/LLMTutorial.git
   cd LLMTutorial
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install --upgrade pip
   pip install torch transformers sentence-transformers faiss-cpu datasets peft accelerate tqdm ipywidgets
   ```
   - Install `faiss-gpu` instead of `faiss-cpu` if you have CUDA available.
   - The notebooks rely on `ipywidgets` for rich progress bars; without it you may see an `IProgress not found` warning when running cells.

4. **Authenticate with Hugging Face** (required to download Gemma checkpoints):
   ```bash
   huggingface-cli login
   ```

5. **Launch Jupyter**:
   ```bash
   jupyter lab  # or: jupyter notebook
   ```

## Notebook Walkthroughs

### 1. LLMInference_demo.ipynb

This notebook demonstrates how to perform chat-style inference with `gemma-3-4b-it` from Hugging Face Transformers.

Key steps:
1. **Model loading** – `create_model()` downloads the tokenizer and model, sends them to GPU when available, and enables evaluation mode for deterministic generation. The `device_map="auto"` argument automatically shards the model across available devices.
2. **Prompt templating** – `lm_template()` wraps user input inside a system/user message structure compatible with chat-optimized models.
3. **Generation loop** – `generate()` applies the chat template, moves tensors to the correct device, and calls `model.generate()` with adjustable `max_new_tokens` and `temperature` parameters.
4. **Test cases** – A short list of sample prompts (small talk, arithmetic, weather questions) prints both the user query and the generated response so you can validate end-to-end inference quickly.

You can adapt the demo by swapping the `lm_model_name`, adjusting decoding parameters, or feeding conversational history instead of single-turn prompts.

### 2. funcCalling_demo.ipynb

Extends the base inference pipeline with a lightweight tool-use agent.

Highlights:
1. **Tool registry** – Defines a calculator tool with structured metadata (`name`, `description`, `parameters`, and callable `function`). Additional tools can be added to `TOOL_DICT` following the same schema.
2. **Structured system prompt** – `build_system_prompt_with_tools()` enumerates the available tools and appends a schema instructing the model to produce `<function_call>{...}</function_call>` tags whenever it wants to call a tool.
3. **Agent loop** – `run_agent()` iteratively queries the model, parses any returned function call JSON with `parse_function_call()`, executes the tool via `execute_tool()`, and feeds the result back into the model until a natural-language answer is produced or the iteration limit is reached.
4. **Post-processing** – `filter_function_call()` removes leftover tool-call markup to return a clean answer.
5. **Demo queries** – The notebook runs several questions that force both tool usage (arithmetic) and direct responses (chit-chat) so you can observe the decision flow.

Use this notebook as a template for integrating APIs, databases, or other deterministic tools into an LLM agent.

### 3. RAG_demo.ipynb

Implements a retrieval-augmented generation workflow combining sentence embeddings, FAISS vector search, and the chat model.

Pipeline overview:
1. **Embedding model** – `create_embedding_model()` loads `all-MiniLM-L6-v2` from Sentence Transformers and prepares it for inference.
2. **Document dataclasses** – `DocChunk` stores each chunk’s text, embedding, and identifiers while `lookupQuery` wraps incoming user questions.
3. **Chunking and indexing** – `preprocess_pages2chunks()` trims input records, `load_doc_from_path()` ingests a JSONL file, creates dense embeddings, and builds a FAISS inner-product index (with L2 normalization for cosine similarity).
4. **Retrieval** – `retrieve_relevant_docs()` searches the index for the top-`k` passages, deduplicates them, and returns rich chunk objects.
5. **Prompt construction** – Retrieved passages are concatenated into a context block that is injected into a prompt template instructing the model to cite references explicitly.
6. **Generation** – `rag_ask()` orchestrates retrieval + generation and returns both the response and the supporting documents.
7. **Demo run** – Sample questions about National Cheng Kung University show how the model uses the retrieved knowledge to answer factual questions more reliably than pure inference.

Before running the notebook, place your own knowledge base at `./dataset/ncku_wikipedia_2510080406.jsonl` (see [Working With Your Own Data](#working-with-your-own-data)).

### 4. LoRAFinetune_demo.ipynb

Showcases parameter-efficient fine-tuning of `google/gemma-3-1b-pt` using Low-Rank Adaptation (LoRA).

Workflow:
1. **Model preparation** – `create_model()` loads the base model in training mode with `attn_implementation="eager"` for compatibility with LoRA fine-tuning.
2. **Dataset ingestion** – `create_dataset()` reads a JSONL corpus, converts it into a Hugging Face `Dataset`, tokenizes text with padding and truncation, masks padded positions in the labels, and splits the result into train and evaluation subsets.
3. **LoRA configuration** – `LoraConfig` targets attention and MLP projection layers (`q_proj`, `k_proj`, `v_proj`, etc.) with rank 16 adapters, alpha 32, and dropout 0.05. `get_peft_model()` wraps the base model so only adapter parameters are trainable.
4. **Training setup** – `TrainingArguments` define 100 epochs, batch sizes of 4 (with gradient accumulation), a 2e-4 learning rate, fused AdamW optimizer, warmup steps, and checkpoint/evaluation cadence. `DataCollatorForLanguageModeling` prepares causal LM batches without masked language modeling.
5. **Execution & saving** – `Trainer.train()` launches fine-tuning, then `trainer.save_model("./gemma-3-1b-pt-lora-final")` persists the adapted weights for downstream use.

Modify the dataset path, LoRA hyperparameters, or training arguments to suit your project. Because only adapter weights are updated, this approach is GPU- and memory-friendly compared with full fine-tuning.

## Working With Your Own Data

Several notebooks expect external resources that are not bundled in the repository:

- **Knowledge base for RAG and LoRA**: Provide a UTF-8 JSONL file at `./dataset/ncku_wikipedia_2510080406.jsonl`. Each line should be a JSON object containing at least a `"text"` field with the content to index or train on. Additional metadata (e.g., titles) can be included and will be preserved in the dataset even if not directly used.
- **Custom corpora**: To use a different file path, update the `documents_path` variable in the relevant notebook cells. Ensure the JSONL structure matches what `create_dataset()` and `preprocess_pages2chunks()` expect (i.e., a flat dictionary where `text` holds the body content).
- **New tools**: In the function-calling demo, add new tool definitions to `TOOL_DICT` by following the calculator example—include a callable `function`, human-readable `description`, and a JSON schema under `parameters`.

## Troubleshooting

- **`IProgress not found` warning**: Install `ipywidgets` and enable widgets in Jupyter. The warning does not stop execution but progress bars will be missing otherwise.
- **Model download errors**: Confirm you are logged into Hugging Face, have accepted the Gemma model licenses, and are using a sufficiently new version of `transformers` (v4.38 or later is recommended for the Gemma 3 family).
- **CUDA out of memory**: Reduce `max_new_tokens`, switch to a smaller checkpoint, enable 8-bit loading (e.g., `load_in_8bit=True` in `from_pretrained`), or run on CPU for experimentation.
- **FAISS import failures**: Install the correct FAISS package for your platform (`faiss-cpu` for most environments, `faiss-gpu` if CUDA is available).
- **Training instability**: Adjust `learning_rate`, `num_train_epochs`, or `gradient_accumulation_steps` in `TrainingArguments`. Monitoring evaluation loss via the built-in validation split helps detect overfitting early.

## License

This project is distributed under the terms of the [MIT License](LICENSE). Feel free to use, modify, and share the notebooks in accordance with that license.