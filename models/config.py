import argparse



def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--dev_data_file", default=None, type=str, required=True, )
    parser.add_argument("--test_data_file", default=None, type=str, required=True, )
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--generation_file", default=None, type=str, required=True)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--source_length", default=16, type=int)
    parser.add_argument("--target_length", default=16, type=int)
    parser.add_argument("--max_length", default=16, type=int)
    parser.add_argument("--num_beams", default=3, type=int)

    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-6, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Linear warmup over warmup_ratio.")

    parser.add_argument('--logging_steps', type=int, default=400,
                        help="Log every X updates steps.")
    parser.add_argument('--validate_steps', type=int, default=3000)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--do_train', action='store_true', help="test")
    parser.add_argument('--do_eval', action='store_true', help="test")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--gpu_id', type=int, default=0, help="For distant debugging.")
    parser.add_argument('--verbose', action='store_true',
                        help="test")
    # seq2seqvae config
    parser.add_argument("--latent_size", default=64, type=int)

    parser.add_argument("--dim_target_kl", default=3.0, type=float,
                        help="dim_target_kl free bit training mode.")
    parser.add_argument("--fb_mode", default=3, type=int)
    parser.add_argument("--beta", type=float, default=1.0,
                        help="The weighting hyper-parameter of the KL term in VAE")
    parser.add_argument("--ratio_increase", default=0.25, type=float,
                        help="Learning schedule, the percentage for the annealing stage.")
    parser.add_argument("--ratio_zero", default=0.25, type=float,
                        help="Learning schedule, the percentage for the pure auto-encoding stage.")

    parser.add_argument("--use_deterministic_connect", default=0, type=int)

    parser.add_argument("--latent_embed_src", default=0, type=int)
    parser.add_argument("--latent_embed_trg", default=0, type=int)
    parser.add_argument("--latent_memory", default=0, type=int)
    parser.add_argument("--classifier_path", default='', type=str)
    parser.add_argument("--lambda_clas", default=0.0, type=float)
    parser.add_argument("--lambda_cx", default=0.1, type=float)

    parser.add_argument("--lambda_reg_z", default=0.0, type=float)

    parser.add_argument("--gumbel_temperature", default=0.0, type=float)
    parser.add_argument("--start_cla_epoch", default=0, type=int)
    parser.add_argument("--eval_ddp", default=0, type=int)
    parser.add_argument("--max_cptlen", default=300, type=int)
    parser.add_argument("--max_trilen", default=1000, type=int)
    parser.add_argument("--evtkg_lambda", default=0.5, type=float)
    parser.add_argument("--topk", default=3, type=int)
    parser.add_argument("--hop_num", default=2, type=int)
    parser.add_argument("--gamma", default=0.5, type=float)
    parser.add_argument("--evtneg_weight", default=0.5, type=float)



    args = parser.parse_args()

    return args
