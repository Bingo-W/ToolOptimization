export DATASET=tooltest_G1_query

python -m new_metrics.CAPR \
    --dataset ${DATASET} \
    --query_path solvable_queries/tool_test_instruction/${DATASET}.json \
    --methods tool_test_dfs_iter_0 \
    --limit 10 \
    --limit_max 5