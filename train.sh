# out_path="./log/normal_ad"
# mkdir $out_path
# export CUDA_VISIBLE_DEVICES=0,1
# nohup python3 main.py --model cnn_adv --train_mode normal_ad --out_path $out_path > $out_path/nohup.log 2>&1 &

# out_path="./log/set_radius_ad_4_0.25"
# mkdir $out_path
# export CUDA_VISIBLE_DEVICES=1
# nohup python3 main.py --model cnn_adv --test_attack_iters 4 --test_attack_step_size 0.25 --train_attack_iters 4 --train_attack_step_size 0.25 --train_mode set_radius_ad --out_path $out_path > $out_path/nohup.log 2>&1 &

# out_path="./log/set_radius_ad_w_1-0-1"
# mkdir $out_path
# export CUDA_VISIBLE_DEVICES=3
# nohup python3 main.py --weight_clean 0 --model cnn_adv --train_mode set_radius_ad --out_path $out_path > $out_path/nohup.log 2>&1 &

# out_path="./log/set_radius_ad_w_1-0-1"
# mkdir $out_path
# export CUDA_VISIBLE_DEVICES=2
# nohup python3 main.py --weight_clean 0 --model cnn_adv --train_mode set_radius_ad --out_path $out_path > $out_path/nohup.log 2>&1 &

# out_path="./log/0312_set_radius_ad_w_0-0-1_sgd"
# mkdir $out_path
# export CUDA_VISIBLE_DEVICES=1
# nohup python3 main.py --weight_adv 0 --weight_clean 0 --model cnn_adv --train_mode set_radius_ad --out_path $out_path > $out_path/nohup.log 2>&1 &

out_path="./log/0312_set_radius_ad_w_0-1-10_sgd"
mkdir $out_path
export CUDA_VISIBLE_DEVICES=1
nohup python3 main.py --weight_adv 0 --weight_clean 1 --weight_ball 10 --model cnn_adv --train_mode set_radius_ad --out_path $out_path > $out_path/nohup.log 2>&1 &

out_path="./log/0312_set_radius_ad_w_0-1-100_sgd"
mkdir $out_path
export CUDA_VISIBLE_DEVICES=2
nohup python3 main.py --weight_adv 0 --weight_clean 1 --weight_ball 100 --model cnn_adv --train_mode set_radius_ad --out_path $out_path > $out_path/nohup.log 2>&1 &

out_path="./log/0312_set_radius_ad_w_0-1-50_sgd"
mkdir $out_path
export CUDA_VISIBLE_DEVICES=3
nohup python3 main.py --weight_adv 0 --weight_clean 1 --weight_ball 50 --model cnn_adv --train_mode set_radius_ad --out_path $out_path > $out_path/nohup.log 2>&1 &