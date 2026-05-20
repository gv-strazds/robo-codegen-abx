#!/bin/bash
# Temporary script to run all 8 updated is_vertical tasks in IsaacSim with teleport + auto-exit

# runs 73 71 26 86 99
# SEEDS=(
#     57407 79107 87218 77987
#     58910 82019 5074 21972
#     59458 57217 18116 54145
#     37018 60182 17725 21972
#     13409 41046 83371 64378
# )
SEEDS=(
1522
46575
30198
55783
44707
9629
87214
88030
92205
29986
96875
88287
57170
94738
45955
70302
68311
43086
35635
50117
)

TASKS=(
    TableTaskBottlesToConveyor2
)
    # TableTaskBottles1
    # TableTaskBottlesToConveyor
    # TableTaskSoupCans1
    # TableTaskSoupCanPacking
    # TableTaskMixedPacking
    # TableTaskCartToConveyor
    # TableTaskMixedCircle

for task in "${TASKS[@]}"; do
for seed in "${SEEDS[@]}"; do
    echo "========================================"
    echo "Running: $task with seed $seed"
    echo "========================================"
    python run_task.py --task "$task" --target-count 8 -x --seed "$seed" >> ./trial_tasks.out 2>> trial_tasks.stderr
    read -t 1 -p "..."
    echo ""
done
done
