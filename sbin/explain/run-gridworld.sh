#!/bin/bash
set -eo

dir=$(dirname $0)
cd ${dir}/../../

SCRIPT=src/rlplg_experiments/explain/counterfactual_learning.py
NOW=$(date +%s)

echo "Timestamp: $NOW"

for GRID in gridworld_cliff_03 gridworld_cliff_05 gridworld_cliff_07; do
    for BLACKBOX in random q-learning; do
        for CONTROL_FN in q-learning; do
            for REWARD_FN in same-fn counterfactual-reward-fn; do
                for INTERVENTION_PENALTY in on off; do
                    python $SCRIPT \
                        --problem gridworld \
                        --grid-path $HOME/Code/rlplg/assets/env/gridworld/levels/$GRID.txt \
                        --output-path $HOME/fs/experiments/explain-rl/gridworld/$GRID/blackbox_$BLACKBOX/control-fn_$CONTROL_FN/reward-fn_$REWARD_FN/intervention-penalty_$INTERVENTION_PENALTY/$NOW \
                        --blackbox $BLACKBOX \
                        --control-fn $CONTROL_FN \
                        --reward-fn $REWARD_FN \
                        --intervention-penalty $INTERVENTION_PENALTY \
                        --num-episodes 2 &
                done
            done
        done
    done
done
