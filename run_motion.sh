# nohup ./run_motion.sh > tmp/motion_classification.out

# python run_motion.py --model grnn  --num_epoch 20
# python run_motion.py --model rgnn  --num_epoch 20 
# python run_motion.py --model mlp  --num_epoch 10
# python run_motion.py --model lstm  --num_epoch 10
# python run_motion.py --model rnn2gnn  --num_epoch 10
# python run_motion.py --model gnn2rnn  --num_epoch 10

for i in 42 0 1 24 68 92 256 512; do
    python run_motion.py --model grnn  --num_epoch 20 --seed $i
    python run_motion.py --model rgnn  --num_epoch 20 --seed $i
    python run_motion.py --model mlp  --num_epoch 10 --seed $i
    python run_motion.py --model lstm  --num_epoch 10 --seed $i
    python run_motion.py --model rnn2gnn  --num_epoch 10 --seed $i
    python run_motion.py --model gnn2rnn  --num_epoch 10 --seed $i
done
