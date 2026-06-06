if [ ! -d "./logs/TRACE" ]; then
    mkdir -p ./logs/TRACE
fi

model_name=TRACE

seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
train_ratios=(1.0 1.0 1.0 1.0)

d_models=(128 128 128 128)
embed_size=(16 16 16 16)
cuda_ids1=(3 3 3 3)

learning_rate=(5e-3 5e-3 5e-3 5e-3)
dropout=(0.1 0.1 0.1 0.1)

dec_ins=(1024 1024 1024 1024)

for ((i = 0; i < 4; i++))
do

    seq_len=${seq_lens[i]}
    pred_len=${pred_lens[i]}

    export CUDA_VISIBLE_DEVICES=${cuda_ids1[i]}

    python -u run.py \
      --is_training 1 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_${model_name}_${seq_len}_${pred_len} \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len ${seq_len} \
      --pred_len ${pred_len} \
      --enc_in 7 \
      --dec_in ${dec_ins[i]} \
      --c_out 7 \
      --d_model ${d_models[i]} \
      --des 'test trace' \
      --batch_size 64 \
      --learning_rate ${learning_rate[i]} \
      --itr 1 \
      --lossfun_alpha 0.5 \
      --test_batch_size 32 \
      --test_mode 0 \
      --fix_seed 1 \
      --resume_training 0 \
      --resume_epoch 0 \
      --save_every_epoch 0 \
      --use_revin 1 \
      --use_norm 1 \
      --send_mail 0 \
      --save_pdf 0 \
      --train_epochs 30 \
      --patience 5 \
      --lradj type2 \
      --loss_mode L1 \
      --train_ratio 1.0 \
      --dropout ${dropout[i]} \
      --plot_mat_flag 0 \
      --checkpoints ./checkpoints \
      --d_chaos 3 \
      --t_evo 0.1 \
      --num_step 20 \
      --chaos_para_nn 0 \
      --decay_rate 0.5 \
      --coupling_strength 0.1 \
      --invert 0 \
      --chaos_type rossler \
      --derivation rk4 \
      --dec_layers 3 \
      --enc_layers 2 \
      2>&1 | tee -a logs/${model_name}/ETTh1_${model_name}_${seq_len}_${pred_len}.log

done