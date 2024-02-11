# nohup ./run.sh > tmp/covid_v1.out

# python run_covid.py --model grnn  --num_epoch 20
# python run_covid.py --model rgnn  --num_epoch 20
# python run_covid.py --model mlp  --num_epoch 10
# python run_covid.py --model lstm  --num_epoch 10
# python run_covid.py --model rnn2gnn  --num_epoch 10
# python run_covid.py --model gnn2rnn  --num_epoch 10


python run_covid.py --model rgnn --num_epoch 30 --hidden_dim 128 64
python run_covid.py --model rgnn --num_epoch 30 --hidden_dim 128 64 --learning_rate 1e-4
python run_covid.py --model rgnn --num_epoch 30 --hidden_dim 128 128 
python run_covid.py --model rgnn --num_epoch 30 --hidden_dim 128 128 --learning_rate 1e-4