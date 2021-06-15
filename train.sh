echo "Running DATE:" $(date +"%Y-%m-%d %H:%M") 
DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE

# nohup python main.py --env 'SlimeVolley-v0' --hidden-dim 256 --max-frames 20000000 > log/$DATE$RAND.log &
# nohup python main.py --env 'SlimeVolleyNoFrameskip-v0' --hidden-dim 512 --max-frames 30000000 > log/$DATE$RAND.log &
#nohup python main.py --env 'SlimeVolleyNoFrameskip-v0' --train-freq 1000 --batch-size 256 > log/$DATE$RAND.log &
#nohup python main.py --env 'pong_v1' --max-frames 20000000 --ram  > log/$DATE$RAND.log &
# nohup python main.py --env 'pong_v1' --train-freq 1000 --batch-size 256 --eta 1  > log/$DATE$RAND.log &
#python train_dqn_against_baseline.py --env SlimeVolley-v0 --hidden-dim 256 --train-freq 100 --batch-size 256 --max-frames 1000000000 > log/$DATE$RAND.log &
#python train_dqn_against_baseline.py --env SlimeVolleyNoFrameskip-v0 --hidden-dim 512 --max-frames 30000000 > log/$DATE$RAND.log &
# python train_dqn_against_baseline_mp.py --env SlimeVolley-v0 --num-envs 5 --hidden-dim 256 --train-freq 100 --batch-size 1024 --max-frames 500000000 > log/$DATE$RAND.log &
# python train_dqn_against_baseline_mp.py --env SlimeVolley-v0 --gamma 0.99 --eps-decay 1000000 --buffer-size 1000000 --num-envs 10 --hidden-dim 256 --train-freq 10 --batch-size 1024 --max-frames 500000000 > log/$DATE$RAND.log &
# python train_dqn_against_baseline_mp.py --env SlimeVolley-v0 --dueling --eps-final 0.01 --gamma 0.99 --eps-decay 1000000 --buffer-size 1000000 --num-envs 10 --hidden-dim 256 --train-freq 10 --batch-size 1024 --max-frames 500000000 > log/$DATE$RAND.log &

#python nash_dqn.py --env SlimeVolley-v0 --num-envs 5  --save-model $DATE --hidden-dim 256 --evaluation-interval 50000 --train-freq 100 --batch-size 1024 --max-frames 500000000 --eps-final 0.01 --gamma 0.99 --eps-decay 1000000 > log/$DATE$RAND.log &
# python train_dqn_against_baseline.py  --env Pong-ram-v0 --hidden-dim 64  --max-tag-interval 10000   > log/$DATE$RAND.log &
# python nash_dqn.py --env rps_v1 --num-envs 2 --hidden-dim 64 --evaluation-interval 500 --rl-start 1000 > log/$DATE$RAND.log &
# python bilateral_dqn.py --env pong_v1 --num-envs 3 --ram --evaluation-interval 20000 --max-tag-interval 10000 --max-frames 20000000  --hidden-dim 256 > log/$DATE$RAND.log &
#python nash_dqn.py --env pong_v1 --num-envs 5 --ram --save-model $DATE --hidden-dim 256 --evaluation-interval 50000 --max-tag-interval 10000 --train-freq 100 --batch-size 1024 --max-frames 500000000 > log/$DATE$RAND.log &
python nash_dqn.py --env boxing_v1 --num-envs 3 --save-model $DATE --ram --hidden-dim 256 --evaluation-interval 50000 --max-tag-interval 10000 --train-freq 100 --batch-size 1024 --max-frames 500000000 > log/$DATE$RAND.log &
# python nash_double_dqn.py --env pong_v1 --num-envs 3 --ram --hidden-dim 256 --evaluation-interval 50000 --max-tag-interval 10000 --train-freq 100 --batch-size 1024 --max-frames 500000000 > log/$DATE$RAND.log &
