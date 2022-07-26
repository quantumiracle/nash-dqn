DATE=`date '+%Y%m%d_%H%M'`
echo "Save as: " $DATE
mkdir -p log/$DATE

nohup python launch.py --env pettingzoo_boxing_v2 --method nash_dqn --wandb_activate True --wandb_entity quantumiracle >> log/$DATE/boxing_v2_nash_dqn.log &