export TOOLBENCH_KEY=""

export OPENAI_KEY=""
export OPENAI_API_BASE=""
export PYTHONPATH=./
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

export GPT_MODEL="gpt-3.5-turbo-16k"
# export GPT_MODEL="gpt-4o-mini"
export SERVICE_URL="http://localhost:8080/virtual"

# testing on Tool Test Set
for METHOD in CoT@5 DFS_woFilter_w2; do
    for group in tooltest_G1_query tooltest_G2_query tooltest_G3_query; do
    # for group in sub_tooltest_G3_query; do
        if [ $METHOD == "DFS_woFilter_w2" ]; then
            export OUTPUT_DIR="data/answer/tool_test_dfs_iter_0"
        elif [ $METHOD == "CoT@5" ]; then
            export OUTPUT_DIR="data/answer/tool_test_cot_5_iter_0"
        else
            export OUTPUT_DIR="data/answer/tool_test_cot_iter_0"
        fi

        mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/$group

        python toolbench/inference/qa_pipeline.py \
            --tool_root_dir server/tools \
            --backbone_model chatgpt_function \
            --chatgpt_model $GPT_MODEL \
            --openai_key $OPENAI_KEY \
            --max_observation_length 1024 \
            --method $METHOD \
            --input_query_file solvable_queries/tool_test_instruction/${group}.json \
            --output_answer_file $OUTPUT_DIR/$group \
            --toolbench_key $TOOLBENCH_KEY \
            --training \
            --simulated_env \
            --using_optimized_text \
            --optimization_method "ours" \
            --optimization_iteration 0
    done

done

# testing on Agent Test Set
for METHOD in CoT@5 DFS_woFilter_w2; do
    for group in test_G1_query test_G2_query test_G3_query; do
        if [ $METHOD == "DFS_woFilter_w2" ]; then
            export OUTPUT_DIR="data/answer/agent_test_dfs_iter_0"
        elif [ $METHOD == "CoT@5" ]; then
            export OUTPUT_DIR="data/answer/agent_test_cot_5_iter_0"
        else
            export OUTPUT_DIR="data/answer/agent_test_cot_iter_0"
        fi

        mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/$group

        python toolbench/inference/qa_pipeline.py \
            --tool_root_dir server/tools \
            --backbone_model chatgpt_function \
            --chatgpt_model $GPT_MODEL \
            --openai_key $OPENAI_KEY \
            --max_observation_length 1024 \
            --method $METHOD \
            --input_query_file solvable_queries/agent_test_instruction/${group}.json \
            --output_answer_file $OUTPUT_DIR/$group \
            --toolbench_key $TOOLBENCH_KEY \
            --training \
            --simulated_env \
            --using_optimized_text \
            --optimization_method "ours" \
            --optimization_iteration 0
    done

done