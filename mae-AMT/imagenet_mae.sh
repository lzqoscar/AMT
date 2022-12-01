#!/bin/bash
#BSUB -J attn_de8_6015_relast
#BSUB -o %J.out
#BSUB -e %J.err
#BSUB -q gpu_v100
#BSUB  -gpu "num=4:mode=exclusive_process:aff=yes"
module load cuda-11.3
module load gcc-8.2.0
module load anaconda3
source activate
conda deactivate
conda activate cu113
python -m torch.distributed.launch --nproc_per_node=4 /seu_share/home/guijie/220215250/mae_attn_driven_de8/main_pretrain.py \
--output_dir './attnd_decoder8_pretrain_6015_relast' \
--log_dir './attnd_decoder8_pretrain_6015_relast' \
--batch_size 128 \
--accum_iter 8 \
--blr 1.5e-4 \
--resume '/seu_share/home/guijie/220215250/mae-original/mae/output_decoder8_pretrain/checkpoint-40.pth'