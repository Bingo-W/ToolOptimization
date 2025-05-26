cd  toolbench/tooleval
export API_POOL_FILE=../../openai_key.json
export CONVERTED_ANSWER_PATH=../../data/model_predictions_converted
export SAVE_PATH=../../data/pass_rate_results
mkdir -p ${SAVE_PATH}
# export EVAL_MODEL=gpt-3.5-turbo
export EVAL_MODEL=gpt-4o-mini

export OPENAI_KEY=""
export OPENAI_API_BASE=""
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

# running Pass Rate evaluation on Tool Test Set
for METHOD in CoT@5 DFS_woFilter_w2; do
    if [ $METHOD == "DFS_woFilter_w2" ]; then
        export CANDIDATE_MODEL="tool_test_dfs_iter_0"
    elif [ $METHOD == "CoT@5" ]; then
        export CANDIDATE_MODEL="tool_test_cot_5_iter_0"
    else
        export CANDIDATE_MODEL="tool_test_cot_iter_0"
    fi
    mkdir -p ${SAVE_PATH}/${CANDIDATE_MODEL}
    for test_set in tooltest_G1_query tooltest_G2_query tooltest_G3_query; do

        python eval_pass_rate.py \
            --converted_answer_path ${CONVERTED_ANSWER_PATH} \
            --save_path ${SAVE_PATH}/${CANDIDATE_MODEL} \
            --reference_model ${CANDIDATE_MODEL} \
            --test_ids ../../solvable_queries/tool_test_query_ids \
            --max_eval_threads 1 \
            --evaluate_times 3 \
            --test_set  ${test_set}

    done
done

# running Pass Rate evaluation on Agent Test Set
for METHOD in CoT@5 DFS_woFilter_w2; do
    if [ $METHOD == "DFS_woFilter_w2" ]; then
        export CANDIDATE_MODEL="agent_test_dfs_iter_0"
    elif [ $METHOD == "CoT@5" ]; then
        export CANDIDATE_MODEL="agent_test_cot_5_iter_0"
    else
        export CANDIDATE_MODEL="agent_test_cot_iter_0"
    fi
    mkdir -p ${SAVE_PATH}/${CANDIDATE_MODEL}
    for test_set in test_G1_query test_G2_query test_G3_query; do

        python eval_pass_rate.py \
            --converted_answer_path ${CONVERTED_ANSWER_PATH} \
            --save_path ${SAVE_PATH}/${CANDIDATE_MODEL} \
            --reference_model ${CANDIDATE_MODEL} \
            --test_ids ../../solvable_queries/agent_test_query_ids \
            --max_eval_threads 1 \
            --evaluate_times 3 \
            --test_set  ${test_set}

    done
done