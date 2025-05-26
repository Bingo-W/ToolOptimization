#!/bin/bash

cd toolbench/tooleval
export RAW_ANSWER_PATH="../../data/answer"
export CONVERTED_ANSWER_PATH="../../data/model_predictions_converted"

# converting answers for Tool Test Set
for METHOD in CoT@5 DFS_woFilter_w2; do
    # Determine the output path based on the method
    if [ "$METHOD" == "CoT@5" ]; then
        export OUTPUT_PATH="tool_test_cot_5_iter_0"
    elif [ "$METHOD" == "DFS_woFilter_w2" ]; then
        export OUTPUT_PATH="tool_test_dfs_iter_0"
    else
        export OUTPUT_PATH="tool_test_cot_iter_0"
    fi

    # Loop through different test sets
    for test_set in tooltest_G1_query tooltest_G2_query tooltest_G3_query; do
        # Create the output directory if it doesn't exist
        mkdir -p "${CONVERTED_ANSWER_PATH}/${OUTPUT_PATH}"

        # Define the input and output file paths for the current test set
        answer_dir="${RAW_ANSWER_PATH}/${OUTPUT_PATH}/${test_set}"
        output_file="${CONVERTED_ANSWER_PATH}/${OUTPUT_PATH}/${test_set}.json"

        # Convert the raw answers to the desired format
        python convert_to_answer_format.py \
            --answer_dir "${answer_dir}" \
            --method "$METHOD" \
            --output "${output_file}"
    done
done

# converting answers for Agent Test Set
for METHOD in CoT@5 DFS_woFilter_w2; do
    # Determine the output path based on the method
    if [ "$METHOD" == "CoT@5" ]; then
        export OUTPUT_PATH="agent_test_cot_5_iter_0"
    elif [ "$METHOD" == "DFS_woFilter_w2" ]; then
        export OUTPUT_PATH="agent_test_dfs_iter_0"
    else
        export OUTPUT_PATH="agent_test_cot_iter_0"
    fi

    # Loop through different test sets
    for test_set in test_G1_query test_G2_query test_G3_query; do
        # Create the output directory if it doesn't exist
        mkdir -p "${CONVERTED_ANSWER_PATH}/${OUTPUT_PATH}"

        # Define the input and output file paths for the current test set
        answer_dir="${RAW_ANSWER_PATH}/${OUTPUT_PATH}/${test_set}"
        output_file="${CONVERTED_ANSWER_PATH}/${OUTPUT_PATH}/${test_set}.json"

        # Convert the raw answers to the desired format
        python convert_to_answer_format.py \
            --answer_dir "${answer_dir}" \
            --method "$METHOD" \
            --output "${output_file}"
    done
done