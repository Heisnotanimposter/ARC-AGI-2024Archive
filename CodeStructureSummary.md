

# The LLM ARChitect: Solving ARC-AGI Is A Matter of Perspective

![Overview Image](https://raw.githubusercontent.com/da-fr/arc-prize-2024/master/.github/overview.png)

This repository contains the code for **The LLM ARChitect** team's submission to the Kaggle ARC Prize 2024 Contest, which achieved a score of **53.5 points** (56.5 points post-deadline), securing a top position in the competition.

Our approach demonstrates that Large Language Models (LLMs) can effectively tackle the complex abstract reasoning tasks of the Abstraction and Reasoning Corpus (ARC-AGI) by focusing on efficient data modeling, extensive augmentation, and specialized inference techniques.

For an in-depth overview of our methodology, please refer to our full paper:
➡️ **[The LLM ARChitect: Solving ARC-AGI Is A Matter of Perspective](https://da-fr.github.io/arc-prize-2024/the_architects.pdf)**

## 💡 Key Methodological Concepts

Our success is built upon a highly interconnected pipeline with four core innovations:

1.  **Expanded Dataset & Efficient Fine-Tuning:** We significantly expanded the training data using **Re-ARC**, **Concept-ARC**, and **ARC-Heavy** datasets. We employed **LoRA** fine-tuning on models like **Mistral-NeMo-Minitron-8B-Base** and an **ARC-AGI-specific reduced token set** (64 symbols) for dense, transparent data encoding, which minimizes context size and tokenization issues.
2.  **Pervasive Augmentation:** Data augmentation (rotations, transpositions, color permutations, example shuffling) is applied at **training**, **inference**, and **scoring** stages to prevent overfitting, generate diverse candidates, and evaluate solutions from multiple "perspectives."
3.  **Depth-First Search (DFS) Candidate Generation:** Instead of standard greedy or stochastic sampling, we use a custom **DFS** algorithm. This method explores all possible solution paths with a cumulative sampling probability above a specified cutoff (e.g., 5-10%). This guarantees the extraction of the best-scoring solution if it exists within the high-probability space, and is significantly faster and more memory-efficient than beam search.
4.  **Augmentation-Based Candidate Selection (AugScore):** To reliably select the best two candidates for submission, we leverage the model's stability under augmented conditions. We calculate the aggregated log-softmax scores of a candidate across multiple augmented versions of the task (e.g., 8 augmentations). This **AugScore** provides a highly stable and effective metric for filtering incorrect solutions.

## 🛠️ Setup and Execution

### Prerequisites

Our training code relies on the `unsloth` package for efficient fine-tuning. For evaluation, `diskcache` is used for caching inference results.

```bash
# Required packages for local execution
pip install unsloth[cu121] torch diskcache
# Note: Adjust unsloth installation based on your CUDA version.
```

### 1. Initial Fine-Tuning

The initial fine-tuning process prepares the base model using the expanded datasets.

*   **Target Script:** `run_finetuning_[model].py` (e.g., `run_finetuning_Nemo-full.py` for the winning model's base).
*   **Data Requirement:** The datasets (ARC-AGI, Re-ARC, Concept-ARC, ARC-Heavy) must be placed in the designated input folder (details are inside the script).
*   **Execution:**
    ```bash
    python training_code/run_finetuning_Nemo-full.py
    ```
*   **Winning Model:** The trained model is available on Hugging Face: **[Mistral-NeMo-Minitron-8B-ARChitects-Full-bnb-4bit](https://huggingface.co/da-fr/Mistral-NeMo-Minitron-8B-ARChitects-Full-bnb-4bit)**

### 2. Evaluation and Submission Simulation

This step simulates the Kaggle submission process, including optional test-time-training, inference, and final candidate selection.

*   **Target Script:** `run_evaluation_[model].py` (e.g., `run_evaluation_Llama-rearc_with_ttt.py`).
*   **Steps:**
    1.  Load the fine-tuned model.
    2.  (Optional) Perform secondary fine-tuning (Test-Time-Training) on evaluation examples.
    3.  Run the custom **DFS inference** on augmented tasks to generate candidates.
    4.  Apply the **AugScore** selection strategy.
    5.  Generate the final `submission.json` file.
*   **Execution:**
    ```bash
    python training_code/run_evaluation_Llama-rearc_with_ttt.py
    ```

### Kaggle Notebooks

Our original submission notebooks are provided for reference:

*   `arc-prize-2024_kaggle.ipynb`: The original Kaggle submission notebook scoring **53.5 points**. This version uses an offline dataset of Python wheels due to competition constraints.
*   `arc-prize-2024_updated.ipynb`: An updated version that can download packages directly and is suitable for local Jupyter execution (requires `unsloth` installed).
*   **Kaggle Link:** The original submission notebook is also available directly **[on Kaggle](https://www.kaggle.com/code/dfranzen/arc-prize-2024-solution-by-the-architects)**.

### Hardware Note

All models were trained on a single **Nvidia H100 GPU**. If you encounter memory issues, we recommend reducing the `batch_size` and/or the `max_tokens` value. A batch size of `2` should allow fine-tuning `Mistral-NeMo-Minitron-8B-Base` on GPUs with 24 GB memory.

## 📂 Detailed Code Structure

The repository is organized into core utility modules and the main executable scripts for training and evaluation.

| File Name | Purpose | Key Capabilities (Integrating Paper Concepts) |
| :--- | :--- | :--- |
| `arc_loader.py` | **Data Formatting and Loading** | Defines the `ArcDataset` class. Handles building datasets from **Re-ARC, Concept-ARC, ARC-Heavy**. Implements **Augmentation** (modifying, shuffling) and prepares data for the **Reduced Token Set** (64 symbols) tokenization. |
| `model_tools.py` | **Model Utilities** | Contains code to load/save Model and LoRA adapters. Implements logic to **Shrink the Tokenizer and Embedding Layers** for the ARC-AGI-specific token set. Includes a custom Data Collator for masking task inputs and the first output during training. |
| `inference_tools.py` | **Inference Engine** | Implements the core Inference code, including the custom **Depth-First Search (DFS)** algorithm for generating diverse candidates with a probability cutoff. Also performs score calculation. |
| `selection.py` | **Candidate Selector (AugScore)** | Contains various score aggregation methods (like `sum_aug_prob`) to rank candidates. The `EvalTool` class manages the **AugScore** process: calculating aggregated log-softmax scores over multiple augmentations to select the best two answers. |
| `run_finetuning_*.py` | **Initial Fine-Tuning** | **Steps:** 1) Load base model and reduce embedding size. 2) Load and **Augment** training data (Re-ARC, etc.). 3) Create a **LoRA** adapter and execute training. 4) Save the LoRA adapter and merge the model. |
| `run_evaluation_*.py` | **Evaluation and Submission** | **Steps:** 1) Load the fine-tuned model. 2) Optionally perform **Secondary Fine-Tuning (Test-Time-Training)**. 3) Run **DFS inference** on the evaluation set. 4) Apply **AugScore** selection. 5) Write and verify the final `submission.json` file. |


## 1. Environment and Setup

The solution relies heavily on the **Unsloth** library for efficient 4-bit fine-tuning and the **Diskcache** library for caching inference results.

### Key Dependencies

| Library | Purpose |
| :--- | :--- |
| `unsloth` | Efficient 4-bit LoRA fine-tuning and model loading. |
| `diskcache` | Caching of inference results to speed up repeated runs. |
| `datasets` | Handling of training data. |
| `torch`, `numpy` | Core numerical and deep learning operations. |

### Execution Workflow

The tuning process generally follows this flow, reflected in the `run_*.py` scripts:

1.  **Data Preparation:** Load and augment datasets (`arc_loader.py`).
2.  **Model Loading & Tokenization:** Load the base LLM and apply custom tokenization (`model_tools.py`).
3.  **Training (Optional TTT):** Apply LoRA fine-tuning (`run_finetuning_*.py`).
4.  **Inference:** Run the custom DFS sampling (`inference_tools.py`).
5.  **Selection:** Score candidates using AugScore and select the final submission (`selection.py`).

---

## 2. Data Modeling and Tokenization Tuning

The model's "perspective" is defined by how the ARC-AGI grids are tokenized and presented.

### A. Data Formatting (`fmt_opts` in `run_*.py` scripts)

This dictionary defines the string format the LLM sees. Any change here requires retraining the model.

| Parameter | File | Purpose & Tuning Notes |
| :--- | :--- | :--- |
| `preprompt` | `run_*.py` | A string of single-character tokens used as a computational buffer (as noted in the paper, Section 3.2). **Tuning:** Experiment with different sequences or lengths. |
| `query_beg` | `run_*.py` | Token signaling the start of the input grid (e.g., `'I'`). |
| `reply_beg` | `run_*.py` | Token signaling the start of the output grid (e.g., `'\n+/-=O'`). **Tuning:** This is a key separator. Adding more unique tokens here might give the model more "time" to process the input before generating the output. |
| `reply_end` | `run_*.py` | Token signaling the end of the output grid (e.g., `'\n' + tokenizer.eos_token`). |
| `lines_sep` | `run_*.py` | Token separating grid rows (e.g., `'\n'`). |
| `max_tokens` | `run_*.py` | Maximum sequence length for training/inference. **Tuning:** Limited by GPU memory (e.g., 8192 for the 8B model). |

### B. Custom Tokenization (`model_tools.py`)

The solution uses a reduced token set (64 symbols) to avoid standard LLM tokenization issues (grouped-digit tokens).

| Function | File | Purpose & Tuning Notes |
| :--- | :--- | :--- |
| `keep_single_char_tokens` | `model_tools.py` | This function is critical. It shrinks the tokenizer and embedding layers. **Tuning:** If you switch to a new base model, the `keep_tok` list (e.g., `list('ABC...') + tokenizer.tokenize('\n')`) must be checked to ensure all necessary single-character tokens (0-9 for colors, separators, preprompt chars) are included. |

---

## 3. Model and Training Tuning

The training is configured in the `run_finetuning_*.py` scripts, primarily using LoRA and Unsloth's `TrainingArguments`.

### A. Model and LoRA Configuration

| Parameter | File | Purpose & Tuning Notes |
| :--- | :--- | :--- |
| `base_model` | `run_*.py` | The LLM to fine-tune (e.g., `nvidia/Mistral-NeMo-Minitron-8B-Base`). **Tuning:** Experiment with different base models (e.g., larger Llama variants) if computational resources allow. |
| `target_modules` | `run_*.py` | The layers to apply LoRA to. **Tuning:** The current setting includes all linear layers and the **input/output embeddings** (`embed_tokens`, `lm_head`), which is crucial for the custom tokenization to work effectively. |
| `r` (LoRA rank) | `run_*.py` | Rank of the LoRA matrices (e.g., `256`). **Tuning:** Higher rank means more expressiveness but more parameters. `256` is high for LoRA, reflecting the complexity of the ARC-AGI task. |
| `lora_alpha` | `run_*.py` | Scaling factor for LoRA weights (e.g., `24`). |

### B. Training Arguments (`TrainingArguments`)

| Parameter | File | Purpose & Tuning Notes |
| :--- | :--- | :--- |
| `num_train_epochs` | `run_*.py` | Number of epochs. **Tuning:** For the preliminary training, this is often set to `1` with a large, augmented dataset. For **Secondary Training (Test-Time-Training)** in `run_evaluation_*.py`, it's often set higher (e.g., `48` epochs in the paper, or `1` epoch on the full evaluation set). |
| `learning_rate` | `run_*.py` | Learning rate for LoRA adapters (e.g., `1e-4`). |
| `embedding_learning_rate` | `run_*.py` | Learning rate for the embedding layers (e.g., `1e-5`). **Tuning:** This is set lower than the main LR to stabilize the training of the new, smaller embedding layer. |
| `data_collator` | `run_*.py` | Uses `InputMaskingDataCollator`. **Tuning:** The `mask_first_n_examples=1` argument ensures the model is not trained to predict the output of the first example, as it has no context for it (as noted in the paper, Section 3.4). |

---

## 4. Augmentation and Inference Tuning

Augmentation is used in both training and inference. Inference uses a custom Depth-First Search (DFS) sampling method.

### A. Augmentation Options (`arc_loader.py` and `run_*.py`)

The `augment_keys` function in `arc_loader.py` defines the transformations.

| Parameter | File | Purpose & Tuning Notes |
| :--- | :--- | :--- |
| `tp` (Transpose) | `run_*.py` | Transposition (swap X and Y axes). `True` for random, `'all'` for all 2 versions. |
| `rt` (Rotate) | `run_*.py` | Rotation. `True` for random, `'all'` for all 4 versions. |
| `perm` (Permute Colors) | `run_*.py` | Random permutation of the 10 colors. |
| `shfl_ex` (Shuffle Examples) | `run_*.py` | Reordering of the input-output example pairs. |
| `train_aug_opts` | `run_finetuning_*.py` | Used to create a large, diverse training set. |
| `infer_aug_opts` | `run_evaluation_*.py` | Used to generate candidates from multiple perspectives (e.g., `tp='all', rt='all', perm=True`). |

### B. DFS Candidate Generation (`inference_tools.py`)

The `dfs` function implements the Depth-First Probability-Guided Sampling.

| Parameter | File | Purpose & Tuning Notes |
| :--- | :--- | :--- |
| `min_prob` | `run_evaluation_*.py` | The minimum cumulative sampling probability for a path to be explored (e.g., `0.1` or `0.05`). **Tuning:** This is the most critical inference parameter. **Lowering `min_prob`** increases the number of candidates (higher chance of finding the correct solution) but significantly increases inference time. **Raising `min_prob`** speeds up inference but risks missing the correct solution. |
| `max_new_tokens` | `inference_tools.py` | The maximum length of the generated output. Calculated automatically based on the largest possible grid size (30x30). |

---

## 5. Scoring and Selection Tuning (AugScore)

The final step is selecting the best 2 candidates from the large set generated by DFS.

### A. Scoring Algorithms (`selection.py`)

The `selection.py` file defines several aggregation methods for the log-softmax scores.

| Algorithm | File | Purpose & Tuning Notes |
| :--- | :--- | :--- |
| `mul_aug_prob` | `selection.py` | **The winning strategy.** Sums the log-probabilities (equivalent to multiplying the probabilities) across all augmented versions. This favors candidates that are stable and high-probability from multiple perspectives. |
| `sum_aug_prob` | `selection.py` | Sums the raw probabilities across all augmented versions. |
| `max_gen_prob` | `selection.py` | Baseline: Uses the highest probability from the original (un-augmented) inference run. |

### B. Candidate Selection (`EvalTool` in `run_evaluation_*.py`)

The `EvalTool` manages the scoring and ranking.

| Parameter | File | Purpose & Tuning Notes |
| :--- | :--- | :--- |
| `n_guesses` | `run_evaluation_*.py` | The number of top-ranked candidates to submit (always `2` for the ARC-AGI competition). |
| `sorting_algo` | `selection.py` | The index of the scoring algorithm to use for the final submission. **Tuning:** In the provided scripts, the default is `-1` (the last one defined in `all_score_algos`), which is `mul_all_prob` or `mul_aug_prob` depending on the run. To explicitly use the winning strategy, you would set this to the index of `mul_aug_prob` or `sum_aug_prob`. |
| `aug_score_opts` | `run_evaluation_*.py` | The augmentation options used *only* for scoring (e.g., `infer_aug_opts`). This must match the augmentations used to generate the candidates. |

### Tuning Strategy Summary

| Goal | Parameter to Tune | Direction | Impact |
| :--- | :--- | :--- | :--- |
| **Increase Candidate Quality** | `min_prob` (DFS cutoff) | Decrease (e.g., `0.1` to `0.05`) | Higher chance of finding correct solution, but much slower inference. |
| **Speed up Inference** | `min_prob` (DFS cutoff) | Increase (e.g., `0.1` to `0.2`) | Faster inference, but risks missing the correct solution. |
| **Improve Generalization** | `train_aug_opts` | Increase diversity (more `perm`, `shfl_ex`) | Better performance on unseen tasks. |
| **Improve Selection Score** | `sorting_algo` (in `EvalTool`) | Test different aggregation methods | Find the best way to rank candidates (e.g., `mul_aug_prob` vs. `sum_aug_prob`). |
| **Use a Larger Model** | `base_model` | Switch to a larger LLM | Higher potential score, but requires more GPU memory and potentially lower batch size/max tokens. |


## The 5-Step Transformation: From Grid to Sequence

The transformation process can be broken down into five distinct, sequential steps, culminating in the final input sequence the LLM receives.

### Step 1: The Raw Data (2D Grid)

The starting point is the raw ARC-AGI task, which consists of a few training examples and one test example. Each example is a pair of grids: an **Input** grid and an **Output** grid.

*   **Format:** A 2D array of integers, where each integer (0-9) represents a specific color.
*   **Example (Conceptual):**
    ```
    Input:  [[1, 1, 0], [1, 0, 0]]
    Output: [[0, 0, 1], [0, 1, 0]]
    ```

### Step 2: Augmentation (Changing the Model's Perspective)

Before the grids are converted into a string, we can optionally apply transformations. This is the core of our "Matter of Perspective" approach. The underlying task rule remains the same, but the model sees the problem from a different angle.

| Transformation | Effect | Purpose |
| :--- | :--- | :--- |
| **Rotation (`rt`)** | Rotates the grid by 90, 180, or 270 degrees. | Helps the model learn rules independent of orientation. |
| **Transposition (`tp`)** | Flips the X and Y axes (rows become columns). | Crucial for tasks involving vertical lines, which are harder for a left-to-right text model. |
| **Color Permutation (`perm`)** | Randomly swaps the color indices (e.g., all `1`s become `5`s). | Forces the model to learn the *relationship* between colors, not the specific color values. |

### Step 3: Grid to Compact String (The Reduced Token Set)

The 2D grid is now converted into a 1D string. This step is where we implement our **Reduced Token Set** strategy.

*   **Goal:** Use only single-character tokens for all grid data (0-9 for colors) and separators. This prevents the LLM's tokenizer from grouping digits (e.g., treating '10' as a single token), which would obscure the grid structure.
*   **Process:**
    1.  Each color number (0-9) is represented by its single-character digit.
    2.  The rows of the grid are joined together.
    3.  A special **Line Separator** token (`lines_sep`, typically `\n`) is inserted between each row.

*   **Example (Using `\n` as `lines_sep`):**
    ```
    Raw Grid: [[1, 1, 0], [1, 0, 0]]
    Compact String: "110\n100"
    ```

### Step 4: Structuring the Task (The Input/Output Format)

Next, the compact strings are assembled into the full task sequence using special separator tokens defined in the `fmt_opts` dictionary.

| Token | Example Value | Purpose |
| :--- | :--- | :--- |
| `query_beg` | `'I'` | Marks the start of an **Input** grid. |
| `reply_beg` | `'\n+/-=O'` | Marks the start of an **Output** grid. |
| `reply_end` | `'\n<eos>'` | Marks the end of a complete **Example** or the final **Solution**. |

#### A Training Example is formatted as:

$$
\text{Training Example} = [\text{query\_beg}] \text{Input Grid} [\text{reply\_beg}] \text{Output Grid} [\text{reply\_end}]
$$

### Step 5: Final Sequence Assembly (The Full Prompt)

The final sequence is the concatenation of all components, forming the complete prompt that is fed into the LLM.

1.  **Preprompt:** A string of unique characters (e.g., `'ABCDEFGH...'`) used as a computational buffer. The model is trained to ignore this but uses it as a "warm-up" space.
2.  **Training Examples:** All training pairs, formatted as in Step 4.
3.  **Test Query:** The final input grid for which the model must generate the solution.

#### The Final Sequence Structure:

$$
\text{Final Sequence} = [\text{Preprompt}] + \sum (\text{Training Examples}) + [\text{query\_beg}] \text{Test Input Grid} [\text{reply\_beg}]
$$

The LLM's job is to complete this sequence by generating the **Test Output Grid** followed by the **reply\_end** token.

**What the LLM sees (Conceptual Example):**

```
[Preprompt]
ABCDEF...

[Training Example 1]
I110\n100\n+/-=O001\n010\n<eos>

[Training Example 2]
I222\n202\n+/-=O000\n020\n<eos>

[Test Query - The part the model must complete]
I330\n300\n+/-=O  <-- Model starts generating here
```
