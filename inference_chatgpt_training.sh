#!/bin/bash

# --- Environment Variables ---
export TOOLBENCH_KEY="test"
export OPENAI_KEY=""
export OPENAI_API_BASE=""
export PYTHONPATH=./

# --- Unset Proxies ---
unset HTTP_PROXY
unset HTTPS_PROXY
unset http_proxy
unset https_proxy

# --- Model and Service Configuration ---
export GPT_MODEL="gpt-3.5-turbo-16k"
export SERVICE_URL="http://localhost:8080/virtual"

# --- Inference Settings ---
METHOD="CoT@1"
export OUTPUT_DIR="data/answer/train_virtual_chatgpt_cot"
group="trainset"

# --- Create Output Directories ---
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/$group" # Corrected typo: groudp -> group

# --- Run QA Pipeline ---
python toolbench/inference/qa_pipeline.py \
    --tool_root_dir server/tools \
    --backbone_model chatgpt_function \
    --chatgpt_model "$GPT_MODEL" \
    --openai_key "$OPENAI_KEY" \
    --max_observation_length 1024 \
    --method "$METHOD" \
    --input_query_file "solvable_queries/training_instruction/${group}.json" \
    --output_answer_file "$OUTPUT_DIR/$group" \
    --toolbench_key "$TOOLBENCH_KEY" \
    --simulated_env \
    --simulated_type 'task' \
    --training

# --- Evaluation Setup ---
cd toolbench/tooleval

export RAW_ANSWER_PATH="../../data/answer"
export CONVERTED_ANSWER_PATH="../../data/model_predictions_converted"
export MODEL_NAME="virtual_chatgpt_dfs"
export OUTPUT_PATH="train_virtual_chatgpt_cot"

# --- Create Converted Answer Directory ---
mkdir -p "${CONVERTED_ANSWER_PATH}/${OUTPUT_PATH}"

# --- Define Input and Output Paths for Conversion ---
answer_dir="${RAW_ANSWER_PATH}/${OUTPUT_PATH}/${group}"
output_file="${CONVERTED_ANSWER_PATH}/${OUTPUT_PATH}/${group}.json"

# --- Convert to Answer Format ---
python convert_to_answer_format.py \
    --answer_dir "${answer_dir}" \
    --method "$METHOD" \
    --output "${output_file}"