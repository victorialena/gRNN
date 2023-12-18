# nohup ./run.sh > baselines.out

# python run_motion.py --model mlp --hidden_dim 64 64
# python run_motion.py --model lstm --hidden_dim 64 64
# python run_motion.py --model rnn2gnn --hidden_dim 64 64 64 64 64
# python run_motion.py --model gnn2rnn --hidden_dim 64 64 64 64 64

# python run_motion.py --model rgnn --hidden_dim 64 --num_epoch 200
# python run_motion.py --model rgnn --hidden_dim 64 --num_epoch 200 --learning_rate 0.01
# python run_motion.py --model rgnn --hidden_dim 64 --num_epoch 200 --batch_size 256

python run_motion.py --model grnn --hidden_dim 64 64 --num_epoch 200
python run_motion.py --model grnn --hidden_dim 64 64 --num_epoch 200 --learning_rate 0.01
python run_motion.py --model grnn --hidden_dim 64 64 --num_epoch 200 --batch_size 256