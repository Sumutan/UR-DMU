echo start process!
set -x # -e stop when any code raise error; -x print command
start_time=$(date +%s)

# 3.2
CUDA_VISIBLE_DEVICES=0 python ucf_main.py --seed 0 --list_folder ./list/SVMAE/SVMAE_woLoss_CLIP-B --output_path outputs/ucf_8-26_8-22_SVMAE_wo-loss_CLIP-B_seed0 --len_feature 1280 &
CUDA_VISIBLE_DEVICES=1 python ucf_main.py --seed 228 --list_folder ./list/SVMAE/SVMAE_woLoss_CLIP-B --output_path outputs/ucf_8-26_8-22_SVMAE_wo-loss_CLIP-B_seed0 --len_feature 1280 &
CUDA_VISIBLE_DEVICES=2 python ucf_main.py --seed 3407 --list_folder ./list/SVMAE/SVMAE_woLoss_CLIP-B --output_path outputs/ucf_8-26_8-22_SVMAE_wo-loss_CLIP-B_seed0 --len_feature 1280
wait
echo "3.2 ucf_8-26_8-22_SVMAE_wo-loss_CLIP-B Run finish!"

#python ucf_main.py --seed 0 --dataset ucfg1 --batch_size 128 --list_folder ./list/UCFg1_CLIP-B --output_path outputs/ucf_UCFg1_CLIP-B_Batch128_seed0 --len_feature 512
#python ucf_main.py --seed 228 --dataset ucfg1 --batch_size 128 --list_folder ./list/UCFg1_CLIP-B --output_path outputs/ucf_UCFg1_CLIP-B_Batch128_seed228 --len_feature 512
#python ucf_main.py --seed 3407 --dataset ucfg1 --batch_size 128 --list_folder ./list/UCFg1_CLIP-B --output_path outputs/ucf_UCFg1_CLIP-B_Batch128_seed3407 --len_feature 512

# 12.24
#python ucf_main.py --seed 0 --dataset ucfg1 --list_folder ./list/UCFg1_CLIP-B --output_path outputs/ucf_UCFg1_CLIP-B_seed0 --len_feature 512
#python ucf_main.py --seed 228 --dataset ucfg1 --list_folder ./list/UCFg1_CLIP-B --output_path outputs/ucf_UCFg1_CLIP-B_seed228 --len_feature 512
#python ucf_main.py --seed 3407 --dataset ucfg1 --list_folder ./list/UCFg1_CLIP-B --output_path outputs/ucf_UCFg1_CLIP-B_seed3407 --len_feature 512

# 12.23
#python ucf_main.py --seed 0 --list_folder ./list/UCF_CLIP-B --output_path outputs/ucf_UCF_CLIP-B_seed0 --len_feature 512
#python ucf_main.py --seed 228 --list_folder ./list/UCF_CLIP-B --output_path outputs/ucf_UCF_CLIP-B_seed228 --len_feature 512
#python ucf_main.py --seed 3407 --list_folder ./list/UCF_CLIP-B --output_path outputs/ucf_UCF_CLIP-B_seed3407 --len_feature 512

#4.22
#python ucf_main.py --list_folder ./list/UCF_SVMAE-L --output_path outputs/ucf_UCF_SVMAE-L --len_feature 1536
#python ucf_main.py --seed 228 --list_folder ./list/UCF_SVMAE-L --output_path outputs/ucf_UCF_SVMAE-L_228 --len_feature 1536
#python ucf_main.py --seed 3407 --list_folder ./list/UCF_SVMAE-L --output_path outputs/ucf_UCF_SVMAE-L_3407 --len_feature 1536

#python ucf_main.py --list_folder ./list/UCF_SVMAE-B --output_path outputs/ucf_UCF_SVMAE-B --len_feature 1280
#python ucf_main.py --seed 228 --list_folder ./list/UCF_SVMAE-B --output_path outputs/ucf_UCF_SVMAE-B_228 --len_feature 1280
#python ucf_main.py --seed 3407 --list_folder ./list/UCF_SVMAE-B --output_path outputs/ucf_UCF_SVMAE-B_3407 --len_feature 1280

#python ucf_main.py --list_folder ./list --output_path outputs/ucf_i3d_official
#python ucf_main.py --seed 228 --list_folder ./list --output_path outputs/ucf_i3d_official_228
#python ucf_main.py --seed 3407 --list_folder ./list --output_path outputs/ucf_i3d_official_3407


#python ucf_main.py --list_folder ./list/UCF_SVMAE-B

end_time=$(date +%s)
runtime_hours=$(((end_time - start_time)/3600))
echo "任务运行时间：$runtime_hours 小时"
date +"%Y-%m-%d %H:%M:%S"