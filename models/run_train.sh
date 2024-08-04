
export ROOT_PATH=data_path
export classifier_model_path=${ROOT_PATH}/train_bartclassfier_cf_data/checkpoint
export datasize=large


export num_epoch=10
export latentsize=128
export max_trilen=110
export max_cptlen=110
export lambda_c=0.1
export lambda_cx=0.5
export topk=4
export regz=1.0

export evtkg_lambda=0.5


export overlap_ratio=0.6
export kept_chain=80
export evtneg_weight=1
export lossclas=0.1
export temp=0.5
export clastart=4


export filename="_twohop_one50_two100_keptchain${kept_chain}_overlapratio${overlap_ratio}"
export DATA_TYPE=ttlarge_overlap${overlap_ratio}_kept${kept_chain}_evtkgvaeclas-lambdacx${lambda_cx}-evtkglambda${evtkg_lambda}-topk${topk}-evtnegweight${evtneg_weight}-clas${lossclas}-temp${temp}-clastart${clastart}_reproduce_saveallmodel
echo ${DATA_TYPE}

nohup accelerate launch --main_process_port 20688 run_seq2seq.py \
--train_data_file ${ROOT_PATH}/eventkg_bart_vae/train_supervised_${datasize}_original_end_split.json${filename} \
--dev_data_file ${ROOT_PATH}/eventkg_bart_vae/dev_data_original_end_splitted.json${filename} \
--test_data_file ${ROOT_PATH}/eventkg_bart_vae/test_data_original_end_splitted.json${filename} \
--classifier_path ${classifier_model_path} \
--generation_file ${DATA_TYPE}.txt \
--output_dir ${ROOT_PATH}/${DATA_TYPE}/ \
--model_name_or_path 'facebook/bart-base' \
--source_length 200 \
--target_length 96 \
--max_cptlen ${max_cptlen} \
--max_trilen ${max_trilen} \
--evtkg_lambda ${evtkg_lambda} \
--evtneg_weight ${evtneg_weight} \
--topk ${topk} \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 16 \
--num_workers 8 \
--seed 42 \
--overwrite_output_dir \
--num_train_epochs ${num_epoch} \
--learning_rate 5e-5 \
--weight_decay 1e-6 \
--warmup_ratio 0.1 \
--logging_steps 500 \
--validate_steps 1000 \
--gpu_id 7 \
--latent_embed_src 0 \
--latent_embed_trg 0 \
--latent_memory 1 \
--latent_size ${latentsize} \
--beta 1 \
--ratio_zero 0.5 \
--ratio_increase 0.25 \
--fb_mode 1 \
--dim_target_kl 0.5 \
--do_train \
--lambda_cx ${lambda_cx} \
--lambda_clas ${lossclas} \
--lambda_reg_z ${regz} \
--start_cla_epoch ${clastart} \
--gumbel_temperature ${temp} \
--eval_ddp 1 \
> run_${DATA_TYPE}.log 2>&1 &

