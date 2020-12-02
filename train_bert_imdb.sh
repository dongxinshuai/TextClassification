
#out_path="./log_imdb_bert/bert_0-1-0_sp5_it10_lr0.00002_adamw"
#mkdir $out_path
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#nohup python3 bert_textad_imdb.py  --model bert_adv  --batch_size 32 --weight_clean 0 --weight_adv 1 --learning_rate 0.00002 --test_batch_size 32 --weight_kl 0  --out_path $out_path > $out_path/nohup.log 2>&1 &


out_path="./log_imdb_bert/bert_ceadv_1-0-5_lr0.00002_adamw"
mkdir $out_path
export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup python3 bert_textad_imdb.py  --test_attack_iters 10  --model bert_adv  --batch_size 32 --weight_clean 1 --weight_adv 0 --learning_rate 0.00002 --test_batch_size 32 --weight_kl 5 --out_path $out_path > $out_path/nohup.log 2>&1 &


