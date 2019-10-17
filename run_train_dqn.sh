

for setting in $(seq 0 1 0); do sbatch --partition=All train_dqn.sh $setting; done