# nohup ./run_motion.sh > tmp/motion_classification.out

# python run_motion.py --model grnn  --num_epoch 20 
# python run_motion.py --model rgnn  --num_epoch 20 
python run_motion.py --model mlp  --num_epoch 10
python run_motion.py --model lstm  --num_epoch 10
python run_motion.py --model rnn2gnn  --num_epoch 10
python run_motion.py --model gnn2rnn  --num_epoch 10