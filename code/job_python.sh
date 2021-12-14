#!/usr/local_rwth/bin/zsh

#SBATCH --ntasks-per-node=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --job-name=python
#SBATCH --output=log/python-%A-%a.log

echo start all

# start 
# 0-5 
# 0-2 vs 3-5 === learning rate
# 0 vs 1, 3 vs 4 === depth 2 3
# 0 vs 2, 3 vs 5 === max lehgth 8 5

if [ $SLURM_ARRAY_TASK_ID = 0 ];
then 
/home/oo994839/anaconda3/envs/python38/bin/python main.py --job_num=$SLURM_ARRAY_JOB_ID --epoch=60 --depth_of_tag_model=2 --tag_max_length=8
fi

if [ $SLURM_ARRAY_TASK_ID = 1 ];
then 
/home/oo994839/anaconda3/envs/python38/bin/python main.py --job_num=$SLURM_ARRAY_JOB_ID --epoch=60 --tag_max_length=8
fi

if [ $SLURM_ARRAY_TASK_ID = 2 ];
then 
/home/oo994839/anaconda3/envs/python38/bin/python main.py --job_num=$SLURM_ARRAY_JOB_ID --epoch=60 --depth_of_tag_model=2 --tag_max_length=5
fi

if [ $SLURM_ARRAY_TASK_ID = 3 ];
then 
/home/oo994839/anaconda3/envs/python38/bin/python main.py --job_num=$SLURM_ARRAY_JOB_ID --learning_rate=0.0003 --epoch=60 --depth_of_tag_model=2 --tag_max_length=8
fi

if [ $SLURM_ARRAY_TASK_ID = 4 ];
then 
/home/oo994839/anaconda3/envs/python38/bin/python main.py --job_num=$SLURM_ARRAY_JOB_ID --learning_rate=0.0003 --epoch=60 --tag_max_length=8
fi

if [ $SLURM_ARRAY_TASK_ID = 5 ];
then 
/home/oo994839/anaconda3/envs/python38/bin/python main.py --job_num=$SLURM_ARRAY_JOB_ID --learning_rate=0.0003 --epoch=60 --depth_of_tag_model=2 --tag_max_length=5
fi


# 6-9
# === learning rate 08 06 04 02
if [ $SLURM_ARRAY_TASK_ID = 6 ];
then
/home/oo994839/anaconda3/envs/python38/bin/python main.py --job_num=$SLURM_ARRAY_JOB_ID --learning_rate=0.0008 --epoch=60 --depth_of_tag_model=3 
fi

if [ $SLURM_ARRAY_TASK_ID = 7 ];
then
/home/oo994839/anaconda3/envs/python38/bin/python main.py --job_num=$SLURM_ARRAY_JOB_ID --learning_rate=0.0006 --epoch=60 --depth_of_tag_model=3 
fi

if [ $SLURM_ARRAY_TASK_ID = 8 ];
then
/home/oo994839/anaconda3/envs/python38/bin/python main.py --job_num=$SLURM_ARRAY_JOB_ID --learning_rate=0.0004 --epoch=60 --depth_of_tag_model=3 
fi

if [ $SLURM_ARRAY_TASK_ID = 9 ];
then
/home/oo994839/anaconda3/envs/python38/bin/python main.py --job_num=$SLURM_ARRAY_JOB_ID --learning_rate=0.0002 --epoch=60 --depth_of_tag_model=3 
fi


# 10-13
# === depth 2 3 4 5 


echo end all

