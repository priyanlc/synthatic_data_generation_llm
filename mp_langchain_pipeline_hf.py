from __future__ import annotations

import os
import re
import io
import csv
import time
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Iterable, Tuple, Any

from multiprocessing import get_context, Process, Queue


# --- Stage 1: prompt factory using the PromptTemplate ------------------------
from typing import Optional, List
import numpy as np
import pandas as pd

def _format_examples_from_seed(seed_data: pd.DataFrame, n_examples: int = 10) -> str:
    """
    Sample n_examples rows and format as CSV lines in the exact column order.
    """
    
    required_cols = ["age", "workclass", "education", "occupation", "sex", "income"]
    missing = [c for c in required_cols if c not in seed_data.columns]
    if missing:
        raise ValueError(f"seed_data missing required columns: {missing}")
    sample = seed_data.sample(n=n_examples)[required_cols]
    # Convert each row to a CSV line and join with newlines
    return "\n".join(sample.apply(lambda x: ",".join(x.astype(str)), axis=1))


def make_prompts_from_seed(
    seed_data: pd.DataFrame,
    prompt_template,  # LangChain PromptTemplate
    n_batches: int = 30,
    n_examples: int = 10,
    random_state: Optional[int] = None,
) -> List[str]:
    """
    Build a list of prompts using the PromptTemplate and fresh examples per batch.
    """
    rng = np.random.default_rng(random_state)
    prompts: List[str] = []
    for _ in range(n_batches):
        examples = _format_examples_from_seed(seed_data, n_examples=n_examples)
        prompt_text = prompt_template.format(examples=examples)
        prompts.append(prompt_text)
    return prompts

# --------------------------- Messages & Sentinels ---------------------------
"""
These are sentinel values: special constant strings used to mark the end of a queue or stream.
"""
PROMPT_SENTINEL = "__PROMPT_QUEUE_DONE__"
RESULT_SENTINEL = "__RESULT_QUEUE_DONE__"

@dataclass
class PromptMsg:
    """
    Represents a prompt message to be sent for processing (e.g., to an LLM worker).

    Attributes:
        id (int):
            Unique identifier for the prompt message, used to correlate
            results with their originating prompt.
        prompt (str):
            The actual prompt text that should be processed.
        attempts (int, default=0):
            Number of times this prompt has already been attempted.
            Useful for retry logic.
        max_retries (int, default=3):
            Maximum number of times this prompt may be retried before
            being considered failed.
    """
    id: int
    prompt: str
    attempts: int = 0
    max_retries: int = 3

 
@dataclass
class ResultMsg:
    """
    Represents the result of processing a prompt message.

    Attributes:
        id (int):
            Identifier matching the originating PromptMsg.id, so results
            can be linked back to their prompt.
        prompt (str):
            The original prompt text that was processed.
        raw_text (Optional[str]):
            The unprocessed raw output (e.g., from the LLM).
            May be None if generation failed.
        parsed_ok (bool):
            Indicates whether the raw_text was successfully parsed into
            the expected structure.
        sampled_fields (Optional[List[str]]):
            Structured fields extracted from the raw output.
            For example, a list of CSV columns or JSON keys.
        ts (float):
            Timestamp (in seconds since epoch) when the result was created.
        error (Optional[str], default=None):
            Error message if processing or parsing failed, otherwise None.
    """
    id: int
    prompt: str
    raw_text: Optional[str]
    parsed_ok: bool
    sampled_fields: Optional[List[str]]
    ts: float
    error: Optional[str] = None

# --------------------------- Validation & DP helpers ---------------------------

from pydantic import BaseModel, ValidationError, validator

class Row(BaseModel):
    age: int
    workclass: str
    education: str
    occupation: str
    sex: str
    income: str

    @validator("age")  
    def _age_bounds(cls, v: int) -> int:
        if not (18 <= v <= 90):
            raise ValueError("age must be between 18 and 90")
        return v

DEFAULT_DRAFTER: Dict[str, List[str]] = {
    "workclass": ["Private", "Self-emp", "Federal-gov", "Local-gov", "State-gov"],
    "education": ["Bachelors", "HS-grad", "Masters", "Some-college", "Doctorate"],
    "occupation": ["Prof-specialty", "Craft-repair", "Exec-managerial", "Adm-clerical"],
    "sex": ["Male", "Female"],
    "income": ["<=50K", ">50K"],
}

def parse_single_csv_row_to_row(text: str) -> Row:
    """
    Parse a single CSV-formatted row into a validated Row object.

    This function reads CSV content from a string, optionally skips a header row,
    validates the structure, and constructs a `Row` instance using Pydantic.
    The CSV must contain exactly six columns in the following order:

        age, workclass, education, occupation, sex, income

    Args:
        text (str):
            A CSV string containing either:
              - a single data row, or
              - a header line followed by exactly one data row.

    Returns:
        Row:
            A validated Row object with typed fields.

    Raises:
        ValueError:
            - If no CSV content is found.
            - If more than one data row is present.
            - If the number of columns is not exactly six.
        ValidationError:
            Raised by Pydantic if the row fails type checks
            or field-level validation (e.g., age out of bounds).
    """
    reader = csv.reader(io.StringIO(text))
    rows: List[List[str]] = [r for r in reader if r and any(cell.strip() for cell in r)]
    if not rows:
        raise ValueError("No CSV content found")

    expected_header = ["age", "workclass", "education", "occupation", "sex", "income"]
    if [c.strip().lower() for c in rows[0]] == expected_header:
        rows = rows[1:]

    if len(rows) != 1:
        raise ValueError(f"Expected exactly 1 CSV data row, got {len(rows)}")

    fields = [c.strip() for c in rows[0]]
    if len(fields) != 6:
        raise ValueError(f"Expected 6 columns, got {len(fields)}: {fields}")

    return Row(
        age=int(fields[0]),
        workclass=fields[1],
        education=fields[2],
        occupation=fields[3],
        sex=fields[4],
        income=fields[5],
    )

def add_dp_noise_probs(predictions: List[float], epsilon: float = 1.0) -> List[float]:
    """
    Add Laplace noise to a probability distribution for differential privacy.

    Each probability is perturbed by Laplace-distributed noise with scale 1/epsilon,
    then the distribution is clipped to be non-negative and renormalized to sum to 1.
    This preserves a valid probability distribution while providing differential
    privacy protection by masking exact values.

    Args:
        predictions (List[float]):
            Original probability distribution (values should sum to ~1).
        epsilon (float, default=1.0):
            Privacy budget parameter. Higher epsilon adds less noise (weaker privacy),
            while lower epsilon adds more noise (stronger privacy).

    Returns:
        List[float]:
            Noisy probability distribution, normalized to sum to 1.

    Notes:
        - Negative probabilities after noise injection are clipped to zero.
        - If all values are clipped to zero, the function falls back to a uniform distribution.
    """
 
    noise = np.random.laplace(0, 1/epsilon, len(predictions))
    noisy = np.array(predictions, dtype=float) + noise
    noisy = np.clip(noisy, 0, None)
    noisy = (noisy / noisy.sum()) if noisy.sum() > 0 else np.ones_like(noisy) / len(noisy)
    return noisy.tolist()

def dp_sample_row(row: Row, drafter: Dict[str, List[str]]) -> List[str]:
    """
    Generate a differentially private version of a single data row by resampling
    categorical fields with Laplace-noised probabilities.

    The function leaves the numeric `age` field unchanged, but for each categorical
    field (`workclass`, `education`, `occupation`, `sex`, `income`), it:
      1. Builds a uniform probability distribution across the allowed values from `drafter`.
      2. Adds Laplace noise using `add_dp_noise_probs` to create a perturbed distribution.
      3. Randomly samples a replacement value from the allowed options, weighted by
         the noisy probabilities.
    This produces a randomized version of the input row that preserves the schema
    but incorporates differential privacy.

    Args:
        row (Row):
            A validated Row object containing the original data.
        drafter (Dict[str, List[str]]):
            Mapping from column names to their allowed categorical values.
            Typically defined in `DEFAULT_DRAFTER`.

    Returns:
        List[str]:
            A list of string field values representing the new row:
            [age, workclass, education, occupation, sex, income].

    Notes:
        - The `age` field is kept as-is (no noise applied).
        - The categorical fields are resampled independently, so correlations between
          fields are not preserved.
        - Each call produces a randomized output; repeated calls with the same input
          can yield different results.
    """
    fields = [str(row.age), row.workclass, row.education, row.occupation, row.sex, row.income]
    col_names = ["age", "workclass", "education", "occupation", "sex", "income"]
    for i in [1, 2, 3, 4, 5]:
        col = col_names[i]
        options = drafter[col]
        base = [1 / len(options)] * len(options)
        noisy = add_dp_noise_probs(base)
        fields[i] = np.random.choice(options, p=noisy)
    return fields

# --------------------------- HF model builder & caller ---------------------------

def build_chain_hf(
    model_name: str,
    hf_token: Optional[str] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.2,
) -> Any:
    """
    Construct a LangChain-compatible text generation chain using a Hugging Face model.

    This function loads a Hugging Face tokenizer and causal language model,
    initializes a `transformers.pipeline("text-generation")`, and wraps it
    in a `HuggingFacePipeline` so it can be used as a LangChain Runnable.

    It is intended to be called once per worker process (e.g. in a multiprocessing
    setup) so each worker maintains its own model instance.

    Args:
        model_name (str):
            The Hugging Face model identifier (e.g., "gpt2", "meta-llama/Llama-2-7b").
        hf_token (Optional[str], default=None):
            Hugging Face Hub access token, required for private models or gated repos.
        max_new_tokens (int, default=128):
            Maximum number of new tokens to generate for each call.
        temperature (float, default=0.2):
            Sampling temperature controlling randomness. Lower = more deterministic,
            higher = more diverse/creative outputs.

    Returns:
        HuggingFacePipeline:
            A LangChain adapter around the Hugging Face text-generation pipeline.
            Can be directly used in LangChain workflows (chains, agents, Runnables).

    Notes:
        - Uses `device_map="auto"` so the model is automatically placed on available
          hardware (GPUs or CPU).
        - Uses `torch_dtype="auto"` to select the most efficient precision available.
        - Sets `return_full_text=False` so only the generated continuation is returned.
        - With `trust_remote_code=True`, custom model code from Hugging Face Hub is allowed.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from langchain_huggingface import HuggingFacePipeline
    import torch

    tok = AutoTokenizer.from_pretrained(
        model_name, 
        use_fast=True, # uses Rust-backed fast tokenizers if available
        token=hf_token)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # automatically places model on available GPUs/CPU
        torch_dtype="auto", # chooses best dtype (float32/16/bf16 depending on HW)
        token=hf_token,
        trust_remote_code=True, # allows custom model code from HF Hub
    )
    gen = pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=gen)


def call_llm(chain: Any, prompt: str) -> str:
    """
    Invoke a language model or LangChain Runnable with a given prompt,
    handling multiple possible input/output interfaces.

    This function provides a uniform way to call different types of
    LLM wrappers (e.g., Hugging Face pipelines, LangChain Runnables).
    It first checks for an `.invoke()` method (LangChain convention),
    falling back to treating the object as a plain callable if needed.
    Both raw string input and {"input": ...} dictionary formats are tried.

    Args:
        chain (Any):
            The model, pipeline, or LangChain Runnable to invoke.
            Can be one of:
              - A LangChain Runnable (with `.invoke()` method).
              - A Hugging Face `pipeline` or other callable object.
        prompt (str):
            The input prompt string to send to the model.

    Returns:
        str:
            The model output converted to plain text. If the underlying
            call returns a non-string (e.g., dict, list), it is coerced
            into a string with `str()`.

    Behavior:
        1. If `chain.invoke(prompt)` works, return its result.
        2. If not, try `chain.invoke({"input": prompt})`.
        3. If `.invoke` doesn’t exist, try calling `chain(prompt)`.
        4. If that fails, try `chain({"input": prompt})`.
        5. Always return the result as a string.

    Notes:
        - Ensures compatibility with both Hugging Face and LangChain APIs.
        - Useful when building pipelines where the exact model interface
          (Runnable vs callable) may vary.
    """
    inv = getattr(chain, "invoke", None)
    if callable(inv):
        try:
            out = chain.invoke(prompt)
        except Exception:
            out = chain.invoke({"input": prompt})
        return out if isinstance(out, str) else str(out)

    # last resort: treat chain as callable
    try:
        out = chain(prompt)
    except Exception:
        out = chain({"input": prompt})
    return out if isinstance(out, str) else str(out)


def _extract_generated_line(text: str) -> str:
    """
    Extract the inner content of the first Markdown-style fenced code block
    (delimited by triple backticks). If no code block is found, return the
    stripped input text.
    """
    m = re.search(r"```(?:\w+)?\s*(.*?)```", text, flags=re.DOTALL)
    return m.group(1).strip() if m else text.strip()

# --------------------------- CPU postprocess ---------------------------

def cpu_postprocess(csv_line: str, drafter: Dict[str, List[str]]) -> Tuple[bool, Optional[List[str]], Optional[str]]:
    """
    Process a single CSV line by parsing it into a validated Row and applying
    differential privacy sampling to categorical fields.

    Workflow:
        1. Parse the input CSV line into a Pydantic `Row` using
           `parse_single_csv_row_to_row`.
        2. Resample the categorical fields (workclass, education, occupation,
           sex, income) with differential privacy noise via `dp_sample_row`.
        3. Return a tuple indicating whether processing succeeded.

    Args:
        csv_line (str):
            A single row of CSV text (with or without a header).
        drafter (Dict[str, List[str]]):
            Mapping from column names to allowed categorical values.

    Returns:
        Tuple[bool, Optional[List[str]], Optional[str]]:
            - parsed_ok (bool): True if parsing and sampling succeeded, False otherwise.
            - sampled_fields_or_none (Optional[List[str]]): The processed row as a list
              of strings if successful, else None.
            - error_or_none (Optional[str]): Error message if processing failed,
              else None.
    """
    try:
        row = parse_single_csv_row_to_row(csv_line)
        sampled = dp_sample_row(row, drafter)
        return True, sampled, None
    except Exception as e:
        return False, None, repr(e)

# --------------------------- Processes ---------------------------

def producer_proc(prompts: Iterable[str], prompt_q: Queue, num_workers: int) -> None:
    """
    Enqueue prompt messages for worker processes and send termination signals.

    This function pushes all provided prompt strings into a shared multiprocessing
    queue as `PromptMsg` objects, each with a unique ID. After all prompts are
    enqueued, it places one sentinel value (`PROMPT_SENTINEL`) into the queue
    for each worker process, signaling them to stop once the work is complete.

    Args:
        prompts (Iterable[str]):
            The collection of prompt strings to enqueue for processing.
        prompt_q (Queue):
            A multiprocessing queue used to distribute prompts to worker processes.
        num_workers (int):
            The number of worker processes that will consume from the queue;
            determines how many sentinel values are placed in the queue.

    Returns:
        None
    """
    for i, p in enumerate(prompts, start=1):
        prompt_q.put(PromptMsg(id=i, prompt=p))
    for _ in range(num_workers):
        prompt_q.put(PROMPT_SENTINEL)

def worker_proc(
    proc_id: int,
    prompt_q: Queue,
    result_q: Queue,
    drafter: Optional[Dict[str, List[str]]] = None,
    hf_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    hf_token: Optional[str] = None,
    gpu_id: Optional[int] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.2,
) -> None:
    """
    Worker process function for consuming prompts, generating model outputs,
    and pushing results into a shared results queue.

    Each worker:
      1. Optionally pins itself to a given GPU (via CUDA_VISIBLE_DEVICES).
      2. Loads a Hugging Face causal language model (through `build_chain_hf`).
      3. Loops over prompts from the shared prompt queue.
      4. For each prompt:
         - Calls the model with the prompt text (`call_llm`).
         - Extracts a CSV-style line from the output (`_extract_generated_line`).
         - Parses and applies DP sampling (`cpu_postprocess`).
         - Pushes a structured `ResultMsg` into the results queue.
      5. Retries failed prompts up to their `max_retries` limit.
      6. Exits cleanly when the `PROMPT_SENTINEL` is received.

    Args:
        proc_id (int):
            Unique identifier for this worker process.
        prompt_q (Queue):
            Multiprocessing queue containing `PromptMsg` objects and sentinel
            values pushed by the producer.
        result_q (Queue):
            Multiprocessing queue where this worker places `ResultMsg` objects
            with results or errors.
        drafter (Optional[Dict[str, List[str]]], default=None):
            Mapping from column names to allowed categorical values for
            DP resampling. Defaults to `DEFAULT_DRAFTER` if not provided.
        hf_model_name (str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
            Hugging Face model identifier to load.
        hf_token (Optional[str], default=None):
            Hugging Face Hub access token (required for private/gated models).
        gpu_id (Optional[int], default=None):
            If provided, pins this worker to the specified GPU index.
        max_new_tokens (int, default=128):
            Maximum number of tokens to generate for each output.
        temperature (float, default=0.2):
            Sampling temperature controlling randomness in generation.

    Returns:
        None

    Notes:
        - Each worker loads its own model instance, so memory usage scales
          with the number of workers.
        - Prompts that fail are retried up to `PromptMsg.max_retries` times.
        - Results are timestamped and always pushed to the result queue,
          whether successful or failed.
    """
    # (Optional) Pin this worker to a specific GPU
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    chain = build_chain_hf(
        model_name=hf_model_name,
        hf_token=hf_token,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    active_drafter = drafter or DEFAULT_DRAFTER

    while True:
        item = prompt_q.get()
        if item == PROMPT_SENTINEL:
            break

        assert isinstance(item, PromptMsg)
        try:
            raw = call_llm(chain, item.prompt)
            csv_line = _extract_generated_line(str(raw))
            parsed_ok, sampled, err = cpu_postprocess(csv_line, active_drafter)
            result_q.put(ResultMsg(
                id=item.id, prompt=item.prompt, raw_text=csv_line, parsed_ok=parsed_ok,
                sampled_fields=sampled, ts=time.time(), error=err
            ))
        except Exception as e:
            item.attempts += 1
            if item.attempts <= item.max_retries:
                prompt_q.put(item)
            else:
                result_q.put(ResultMsg(
                    id=item.id, prompt=item.prompt, raw_text=None, parsed_ok=False,
                    sampled_fields=None, ts=time.time(), error=repr(e)
                ))

def finalizer_proc(result_q: Queue, outfile_csv: str, outfile_log: Optional[str] = None) -> None:
    """
    Finalizer process that consumes results from a queue and writes them to disk.

    This process continuously reads `ResultMsg` objects from the result queue
    and persists them to output files until it encounters the `RESULT_SENTINEL`.
    Successful rows are appended to a CSV file, and all results (successes and
    failures) can optionally be logged in JSONL format for auditing/debugging.

    Workflow:
        1. Ensure the output directory exists.
        2. Open the CSV file in append mode; write a header row if the file is new.
        3. Optionally open a JSONL log file if `outfile_log` is provided.
        4. Consume results from the queue:
            - Stop when `RESULT_SENTINEL` is received.
            - For successful results (`parsed_ok=True` with sampled fields),
              append the row to the CSV file.
            - For all results, serialize the full `ResultMsg` as JSON and
              append it to the log file if logging is enabled.
        5. Flush data to disk after each write to reduce risk of data loss.

    Args:
        result_q (Queue):
            Multiprocessing queue populated by worker processes with `ResultMsg`
            objects and a final `RESULT_SENTINEL`.
        outfile_csv (str):
            Path to the CSV file where processed rows will be appended.
            A header row is written if the file is new or empty.
        outfile_log (Optional[str], default=None):
            Path to a JSONL log file where every result (including failures)
            is appended. If None, no logging is performed.

    Returns:
        None

    Notes:
        - The process exits cleanly when `RESULT_SENTINEL` is read from the queue.
        - The CSV file only contains successful results, while the log file
          (if enabled) captures the full history, including errors.
        - Uses `flush()` after each write to minimize data loss in case of crash.
    """
    os.makedirs(os.path.dirname(outfile_csv) or ".", exist_ok=True)
    first = not os.path.exists(outfile_csv) or os.path.getsize(outfile_csv) == 0
    hdr = ["age", "workclass", "education", "occupation", "sex", "income"]

    flog = open(outfile_log, "a", encoding="utf-8") if outfile_log else None
    with open(outfile_csv, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if first:
            w.writerow(hdr)

        while True:
            msg = result_q.get()
            if msg == RESULT_SENTINEL:
                break
            assert isinstance(msg, ResultMsg)
            if msg.parsed_ok and msg.sampled_fields:
                w.writerow(msg.sampled_fields)
                f.flush()
            if flog:
                flog.write(json.dumps(asdict(msg), ensure_ascii=False) + "\n")
                flog.flush()
    if flog:
        flog.close()

# --------------------------- Orchestrator ---------------------------
def run_pipeline_multiproc(
    prompts: Iterable[str],
    outfile_csv: str = "synthetic_adult_census_mp.csv",
    outfile_log: Optional[str] = "synthetic_mp_log.jsonl",
    num_workers: int = 1,  # start with 1 for VRAM/RAM; scale only if you have multiple GPUs
    drafter: Optional[Dict[str, List[str]]] = None,
    hf_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    hf_token: Optional[str] = None,
    gpu_ids: Optional[List[int]] = None,   # e.g., [0,1] to pin workers
    max_new_tokens: int = 128,
    temperature: float = 0.2,
) -> None:
    """
    Orchestrate a complete multiprocessing pipeline for prompt-based data generation.

    This function coordinates three types of processes:
      - Producer: pushes all prompts into a shared queue and appends sentinel values
        for each worker.
      - Workers: each worker loads a Hugging Face causal language model, consumes
        prompts, generates outputs, parses them into rows, applies differential privacy
        sampling to categorical fields, and pushes structured results into a results queue.
      - Finalizer: consumes results, appends successful rows to a CSV file, and
        optionally logs all results (including failures) in JSONL format.

    Args:
        prompts (Iterable[str]):
            A collection of prompt strings to be processed.
        outfile_csv (str, default="synthetic_adult_census_mp.csv"):
            Path to the CSV file where successfully parsed and sampled rows
            will be appended. A header is added if the file is new.
        outfile_log (Optional[str], default="synthetic_mp_log.jsonl"):
            Path to a JSONL log file where all results (successes and failures)
            are appended. If None, logging is skipped.
        num_workers (int, default=1):
            Number of worker processes to spawn. Each worker builds its own
            Hugging Face model instance. For large models, start with 1 unless
            you have multiple GPUs.
        drafter (Optional[Dict[str, List[str]]], default=None):
            Mapping of column names to allowed categorical values for
            differential privacy resampling. Defaults to `DEFAULT_DRAFTER`.
        hf_model_name (str, default="meta-llama/Meta-Llama-3.1-8B-Instruct"):
            Hugging Face model identifier to load for generation.
        hf_token (Optional[str], default=None):
            Hugging Face Hub access token, required for private or gated models.
        gpu_ids (Optional[List[int]], default=None):
            List of GPU IDs to pin workers to (e.g., [0, 1] for two GPUs).
            If None, workers run on the default device assignment.
        max_new_tokens (int, default=128):
            Maximum number of tokens to generate per prompt.
        temperature (float, default=0.2):
            Sampling temperature controlling output randomness. Lower is
            more deterministic; higher is more diverse.

    Returns:
        None

    Workflow:
        1. Create multiprocessing queues for prompts and results.
        2. Launch producer, worker(s), and finalizer processes.
        3. Wait for producer and workers to complete.
        4. Send a sentinel to the finalizer to flush remaining results and stop.
        5. Print confirmation when rows are appended to the CSV.

    Notes:
        - Each worker loads its own model instance, so VRAM/RAM usage scales
          with the number of workers.
        - Queues are bounded (maxsize=500) to avoid memory blow-up if producer
          is faster than workers.
        - Finalizer writes successful rows to CSV immediately and flushes to
          disk; JSONL logs capture both successes and errors for auditing.
    """

    ctx = get_context("spawn")
    prompt_q: Queue = ctx.Queue(maxsize=500)
    result_q: Queue = ctx.Queue(maxsize=500)

    procs: List[Process] = []

    # Producer
    p_prod = ctx.Process(target=producer_proc, args=(prompts, prompt_q, num_workers), name="producer")
    procs.append(p_prod)

    # Workers
    for i in range(num_workers):
        gpu_id = None
        if gpu_ids and i < len(gpu_ids):
            gpu_id = gpu_ids[i]
        p = ctx.Process(
            target=worker_proc,
            args=(
                i + 1, prompt_q, result_q, drafter,
                hf_model_name, hf_token, gpu_id, max_new_tokens, temperature
            ),
            name=f"worker-{i+1}",
        )
        procs.append(p)

    # Finalizer
    p_fin = ctx.Process(target=finalizer_proc, args=(result_q, outfile_csv, outfile_log), name="finalizer")
    procs.append(p_fin)

    # Start
    for p in procs:
        p.start()

    # Wait for producer + workers
    p_prod.join()
    for i in range(num_workers):
        procs[1 + i].join()

    # Stop finalizer after draining
    result_q.put(RESULT_SENTINEL)
    p_fin.join()

    print(f"Done. Appended rows to {outfile_csv}")

# --------------------------- putting it all together ---------------------------

def run_pipeline_from_seed(
    seed_data: pd.DataFrame,
    prompt_template,
    n_batches: int = 30,
    n_examples: int = 10,
    random_state: Optional[int] = None,
    *,
    # passthrough to run_pipeline_multiproc
    outfile_csv: str = "synthetic_adult_census_mp.csv",
    outfile_log: Optional[str] = "synthetic_mp_log.jsonl",
    num_workers: int = 1,
    drafter: Optional[Dict[str, List[str]]] = None,
    hf_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    hf_token: Optional[str] = None,
    gpu_ids: Optional[List[int]] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.2,
) -> None:
    """
    End-to-end pipeline for generating synthetic tabular data from a seed dataset.

    This function composes two stages into one call:
      1. Prompt creation (`make_prompts_from_seed`):
         - Randomly sample subsets of rows from the given seed dataset.
         - Insert them into a text prompt template to guide an LLM.
         - Produce a batch of prompts for synthetic row generation.
      2. Multi-process execution (`run_pipeline_multiproc`):
         - Launches a producer, worker processes, and a finalizer.
         - Workers each load a Hugging Face model instance, consume prompts,
           generate candidate rows, parse them into structured form, and apply
           differential privacy sampling to categorical fields.
         - The finalizer writes valid synthetic rows to a CSV file and optionally
           appends all results (success and failure) to a JSONL log.

    Args:
        seed_data (pd.DataFrame):
            Source dataset to draw example rows from when building prompts.
        prompt_template (Union[str, PromptTemplate]):
            Template used to construct prompts. Must contain an {examples}
            placeholder where sampled rows will be inserted.
        n_batches (int, default=30):
            Number of prompts (batches) to generate.
        n_examples (int, default=10):
            Number of seed rows to include in each prompt.
        random_state (Optional[int], default=None):
            Random seed for reproducible sampling of seed data.

        outfile_csv (str, default="synthetic_adult_census_mp.csv"):
            Path to the CSV file where successful synthetic rows are appended.
        outfile_log (Optional[str], default="synthetic_mp_log.jsonl"):
            Path to a JSONL log file capturing all results. If None, logging is disabled.
        num_workers (int, default=1):
            Number of worker processes to spawn. Each worker builds its own
            Hugging Face model instance.
        drafter (Optional[Dict[str, List[str]]], default=None):
            Mapping of column names to allowed categorical values for
            differential privacy resampling. Defaults to `DEFAULT_DRAFTER`.
        hf_model_name (str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
            Hugging Face model identifier to load for generation.
        hf_token (Optional[str], default=None):
            Hugging Face Hub access token, required for private or gated models.
        gpu_ids (Optional[List[int]], default=None):
            List of GPU IDs to pin workers to (e.g., [0, 1]). If None, workers
            use default device assignment.
        max_new_tokens (int, default=128):
            Maximum number of tokens to generate per prompt.
        temperature (float, default=0.2):
            Sampling temperature controlling output randomness. Lower is more
            deterministic; higher is more diverse.

    Returns:
        None

    Notes:
        - Each worker loads its own model instance, so VRAM/RAM usage scales with
          the number of workers.
        - The CSV file only contains successful, parsed synthetic rows. The JSONL
          log (if enabled) contains full results including failures for debugging.
    """
    prompts = make_prompts_from_seed(
        seed_data=seed_data,
        prompt_template=prompt_template,
        n_batches=n_batches,
        n_examples=n_examples,
        random_state=random_state,
    )
    run_pipeline_multiproc(
        prompts=prompts,
        outfile_csv=outfile_csv,
        outfile_log=outfile_log,
        num_workers=num_workers,
        drafter=drafter,
        hf_model_name=hf_model_name,
        hf_token=hf_token,
        gpu_ids=gpu_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


# --------------------------- for testing ---------------------------

def make_simple_prompts(n: int, template: str = "Write a CSV row: age,workclass,education,occupation,sex,income for item {i}") -> List[str]:
    return [template.format(i=i) for i in range(1, n + 1)]

if __name__ == "__main__":
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except Exception:
        pass

    # Demo: small model by default (safe in RAM/VRAM). Swap to your model when ready.
    prompts = make_simple_prompts(10)
    run_pipeline_multiproc(
        prompts,
        num_workers=1,  # keep 1 unless you have multiple GPUs / plenty of RAM
        hf_model_name=os.getenv("HF_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        hf_token=os.getenv("HF_TOKEN"),
        gpu_ids=None,  # e.g., [0,1]
        max_new_tokens=128,
        temperature=0.2,
    )