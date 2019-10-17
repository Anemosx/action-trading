

for step_penalty in $(seq 1 1 2); do sbatch --partition=All train_dqn.sh 0 $step_penalty; done