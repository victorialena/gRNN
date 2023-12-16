# nohup ./run_scripts/eval_ind.sh > run_ood.out

python run_motion.py --model rgnn --hidden_dim 64
python run_motion.py --model grnn --hidden_dim 64 64
python run_motion.py --model mlp --hidden_dim 64 64
python run_motion.py --model lstm --hidden_dim 64 64
python run_motion.py --model rnn2gnn --hidden_dim 64 64 64 64 64
python run_motion.py --model gnn2rnn --hidden_dim 64 64 64 64 64