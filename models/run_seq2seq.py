from __future__ import absolute_import, division, print_function
import argparse
import pickle
import shutil
import time
import re
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import warnings
import json
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from tqdm import tqdm, trange
from transformers import BartTokenizer, RobertaTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup, get_scheduler
from config import get_config
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedType
from accelerate.logging import get_logger
from CausalCRN.tools import set_logger, JsonDumpHelper

from CausalCRN.evaluation.eval_timetravel import load_json_to_instances
import math
from datahelper import RocStoriesDataset, TTEvalDataset
from model_seq2seq import BartVaeSeq2Seq
import numpy as np
from CausalCRN.evaluation.eval_timetravel import (
    eval_bleu, eval_rouge, eval_bert_score, eval_origin_cf_end_bleu
)
from transformers.generation.utils import GenerationMixin
from eval_utils import RobertaClassifier, EntScoreDataset


warnings.filterwarnings('ignore')
logger = get_logger(__name__)

pj_dir = 'datadir/timetravel'
global_ent_path = os.path.join(pj_dir, 'cfstory_nli_metrics/roberta-large')

MODEL_CLASSES = {
    'bart': (BartTokenizer, BartVaeSeq2Seq),

}


gradient_accumulation_steps = 1
accelerator = Accelerator(mixed_precision='fp16', gradient_accumulation_steps=gradient_accumulation_steps)
# Initialize accelerator
if accelerator.distributed_type == DistributedType.TPU and gradient_accumulation_steps > 1:
    raise NotImplementedError(
        "Gradient accumulation on TPUs is currently not supported. Pass `gradient_accumulation_steps=1`"
    )

global_step, tr_loss, logging_loss = 0, 0, 0



def save_model(model, tokenizer, args, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    else:
        os.mkdir(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    pickle.dump(args, open(os.path.join(output_dir, 'args.bin'), 'wb'))
    logger.info('model is saved to {}.\n'.format(output_dir))


def save_generated_samples(samples, output_path):
    with open(output_path, 'w') as fout:
        for d in samples:
            fout.write(json.dumps(d))
            fout.write('\n')


def frange_cycle_zero_linear(n_iter, start=0.0, stop=1.0, n_cycle=4, ratio_increase=0.5, ratio_zero=0.25):
    L = np.ones(n_iter) * stop
    period = n_iter / n_cycle
    step = (stop - start) / (period * ratio_increase)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_iter):
            if i < period * ratio_zero:
                L[int(i + c * period)] = start
            else:
                L[int(i + c * period)] = v
                v += step
            i += 1
    return L


def load_json_samples(path, add_idx=True):
    _samples = []
    idx = 0
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            d = json.loads(line)

            if add_idx:
                d['_sample_idx'] = idx
            _samples.append(d)
            idx += 1
    return _samples



@torch.no_grad()
def eval_ent_nli_ddp(args, instances, entmodel, batch_size=8):
    ent_tokenizer = RobertaTokenizer.from_pretrained(global_ent_path)
    eval_dataset = EntScoreDataset(ent_tokenizer, instances)
    eval_dataloader = DataLoader(
        eval_dataset, shuffle=False, batch_size=batch_size, num_workers=4)
    num_devices = accelerator.num_processes
    batch_num_per_device = int(len(eval_dataset) / (batch_size * num_devices)) + 1
    progress_bar = None
    if args.verbose:
        progress_bar = tqdm(range(batch_num_per_device), disable=not accelerator.is_local_main_process)

    eval_dataloader = accelerator.prepare(eval_dataloader)
    entmodel.eval()
    logger.info(f'start entscore evaluation...')
    scorel = []
    for batch in eval_dataloader:
        input_ids, input_mask = batch
        with torch.no_grad():
            output = entmodel(input_ids,attention_mask=input_mask,)
            logits = output[0]
        logits = accelerator.gather_for_metrics(logits)
        entail_prob = logits.data.detach().softmax(-1)[:, 2].cpu().tolist()  # 2 for entail
        scorel.extend(entail_prob)
        if args.verbose:
            progress_bar.update(1)
    logger.info(f'finished entscore evaluation')

    score = sum(scorel) / len(scorel)
    return {'entail_score': score,}


@torch.no_grad()
def eval_ent_nli_single(instances, nli_scorer, bs):
    tokenizer = RobertaTokenizer.from_pretrained(global_ent_path)
    def score_batch(premises, hypotheses):
        encoded = tokenizer(
            premises, hypotheses, padding="max_length",
            truncation="longest_first", return_tensors="pt",
        )
        for key in encoded.keys():
            encoded[key] = encoded[key].to(nli_scorer.device)
        logits = nli_scorer(**encoded)[0]
        entail_prob = logits.data.detach().softmax(-1)[:, 2].cpu().tolist()  # 2 for entail
        return entail_prob

    i = 0
    scorel = []
    while i < len(instances):
        batch = instances[i:i + bs]
        init_premise_list, coun_premise_list, hypos_list = [], [], []
        for js in batch:
            count_premise = js.cf_context
            hypothesis = js.predicted_ending
            coun_premise_list.append(count_premise)
            hypos_list.append(hypothesis)
        entail_count_prob = score_batch(coun_premise_list, hypos_list)
        scorel.extend(entail_count_prob)
        i += bs
    if len(scorel) == 0: return 0, []
    score = sum(scorel) / len(scorel)
    return {'entail_score': score, }

def do_eval_from_instances(args, instances, metrics, bert_model='bert-base-uncased',
                               entail_model=None, bs=2, is_ddp=True):
    results = {}
    if 'bleu' in metrics:

        results.update(eval_bleu(instances))
        results.update(eval_origin_cf_end_bleu(instances))
    if 'rouge' in metrics:
        results.update(eval_rouge(instances))
        rscore = results.pop('rouge_all')
        results['rouge-l'] = rscore['rouge-l']['f']
    # if 'bertscore' in metrics:
    #     results.update(eval_bert_score(instances, bert_model=bert_model))
    if 'entailscore' in metrics:
        if is_ddp:
            ent_metric = eval_ent_nli_ddp(args, instances, entail_model, batch_size=bs)
        else:
            ent_metric = eval_ent_nli_single(instances, entail_model, bs)
        results.update(ent_metric)
        # results.update(eval_nli(instances, entail_model_path,bs=bs))
        if 'bleu' in metrics:
            bleu = results['corpus_bleu']
            ents = results['entail_score']
            results.update(
                {'hm': (2 * bleu * ents) / (bleu + ents)}
            )
    for k in list(results.keys()):
        if 'instance' in k:
            results.pop(k)
    for k in results:
        results[k] = float('{:.5f}'.format(results[k]*100))
    return results


@torch.no_grad()
def _generation_epoch_for_evaluation(args, model, ent_model, tokenizer, eval_dataset, device, split, epoch):
    model.eval()

    samples = eval_dataset.samples
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.per_gpu_eval_batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False, pin_memory=True
    )
    eval_dataloader = accelerator.prepare(eval_dataloader)
    num_devices = accelerator.num_processes
    batch_num_per_device = int(len(eval_dataset) / (args.per_gpu_eval_batch_size * num_devices)) + 1
    progress_bar = None
    if args.verbose:
        progress_bar = tqdm(range(batch_num_per_device), disable=not accelerator.is_local_main_process)

    max_gen_len = args.target_length
    gens_seq = []
    logger.info(f'start {split} generation....')
    for batch in eval_dataloader:

        src_input_ids, src_attention_mask, \
        decoder_input_ids, decoder_attention_mask, labels, \
            prior_input_ids, prior_attn_mask,\
        post_input_ids, post_attn_mask,\
        trunc_node_list, event_node_label, event_node_distance,\
        head_ids, tail_ids, relation, triple_label, sample_idx = batch
        try:
            fn = model.module
        except:
            fn = model
        with torch.no_grad():
            evt_probs, top_evt_tokens, top_evt_mask = fn.transformer.predicting_guided_event(
                src_input_ids, src_attention_mask,
                trunc_node_list, event_node_label, event_node_distance,
                head_ids, tail_ids, relation, triple_label
            )
        input_ids = torch.cat((src_input_ids, top_evt_tokens), dim=1)
        attention_mask = torch.cat((src_attention_mask, top_evt_mask), dim=1)
        latent_z = None
        use_latent = args.latent_embed_src or args.latent_embed_trg or args.latent_memory
        if use_latent:
            latent_z, z, mu, logvar = fn.sample_latent_z(
                prior_input_ids, prior_attn_mask)

        generated_ids = fn.transformer.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            latent_z=latent_z,
            num_beams=1,
            do_sample=True,
            max_length=max_gen_len,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        generated_ids = accelerator.pad_across_processes(generated_ids, dim=1, pad_index=tokenizer.pad_token_id)
        generated_ids, sample_idx = accelerator.gather_for_metrics((generated_ids, sample_idx))
        sample_idx = sample_idx.cpu().numpy().tolist()
        generated_ids = generated_ids.cpu().numpy().tolist()
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generated_texts = [re.sub('\n', '', text) for text in generated_texts]
        items = list(zip(sample_idx, generated_texts))
        gens_seq.extend(items)
        if args.verbose:
            progress_bar.update(1)

    logger.info('finished {} generation, num-items {}'.format(split, len(gens_seq)))
    for item in gens_seq:
        sample_id, gen_text = item
        samples[sample_id]['generated_endings'] = gen_text.strip()
    save_generated_samples(
        samples,
        os.path.join(args.output_dir, '{}_epoch_{}.jsonl'.format(split, epoch))
    )
    for i in range(2):
        item = samples[i]
        # print('edited_endings: ', item['edited_endings'])
        logger.info('original_ending: {}'.format(item['original_ending']))
        logger.info('generated_endings: {}'.format(item['generated_endings']))
        logger.info('------' * 10)

    metrics = do_eval_from_instances(
        args,
        load_json_to_instances(samples),
        metrics=['bleu', 'rouge', 'entailscore'],
        entail_model=ent_model,
        bs=4,
        is_ddp=args.eval_ddp == 1,

    )

    return metrics, samples



def train(args, model, tokenizer, train_samples, dev_dataset, test_dataset):
    global global_ent_path
    ent_model = RobertaClassifier.from_pretrained(global_ent_path)
    if args.eval_ddp:
        ent_model = accelerator.prepare(ent_model)
    else:
        ent_model = ent_model.to(accelerator.device)

    train_dataset = RocStoriesDataset(
        tokenizer, samples=train_samples, source_length=args.source_length,
        target_length=args.target_length,
        max_cptlen=args.max_cptlen, max_trilen=args.max_trilen, max_evtlen=10,
        is_train=True
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataset) / args.per_gpu_train_batch_size)
    num_training_steps = args.num_train_epochs * num_update_steps_per_epoch
    logger.info(accelerator.state)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.per_gpu_train_batch_size, num_workers=args.num_workers)

    model.set_classifier_eval()
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, cnn.parameters()), lr=learning_rate)

    optimizer = AdamW(
        # model.parameters(),
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay, no_deprecation_warning=True)

    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=int(num_training_steps * args.warmup_ratio),
        num_training_steps=num_training_steps,
    )

    model, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader)

    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' % json.dumps(vars(args), cls=JsonDumpHelper, indent=4, sort_keys=True))
    logger.info('-' * 100)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size, )
    logger.info("  Total optimization steps = %d", num_training_steps)

    num_devices = accelerator.num_processes
    logger.info('num-devices {}'.format(num_devices))
    logger.info("*****==================*****")

    batch_num_per_device = int(len(train_dataset) / (args.per_gpu_train_batch_size * num_devices))+1

    n_iter = int(args.num_train_epochs) * batch_num_per_device
    beta_t_list = frange_cycle_zero_linear(n_iter, start=0.0, stop=args.beta, n_cycle=int(args.num_train_epochs),
                                           ratio_increase=args.ratio_increase, ratio_zero=args.ratio_zero)
    use_latent = args.latent_embed_src or args.latent_embed_trg or args.latent_memory
    start_cla_epoch = args.start_cla_epoch
    # start_cla_epoch =0
    def _train_epoch(epochid):
        global global_step, tr_loss, logging_loss
        progress_bar = None
        if args.verbose:
            progress_bar = tqdm(range(batch_num_per_device), disable=not accelerator.is_local_main_process)

        for step, batch in enumerate(train_dataloader):
            model.train()
            model.module.set_classifier_eval()

            beta_t = beta_t_list[step + epochid * batch_num_per_device]

            if beta_t == 0.0:
                fb_mode = 0
            else:
                fb_mode = 1
            if args.use_deterministic_connect:
                fb_mode = 2

            src_input_ids, src_attention_mask,\
            decoder_input_ids, decoder_attention_mask, labels,\
            prior_input_ids, prior_attn_mask,\
            post_input_ids, post_attn_mask,\
            premise_cond_input_ids, premise_cond_attn_mask,\
            trunc_node_list, event_node_label, event_node_distance,\
            head_ids, tail_ids, relation, triple_label, \
                gumbel_src_input_ids, gumbel_src_attn_mask,\
            gumbel_trg_pos_input_ids, gumbel_trg_pos_attn_mask,\
            gumbel_trg_neg_input_ids, gumbel_trg_neg_attn_mask,\
            clas_pos_input_ids, clas_pos_attn_mask,\
            clas_neg_input_ids, clas_neg_attn_mask = batch

            with accelerator.accumulate(model):
                optimizer.zero_grad()
                outputs = model(
                    input_ids=src_input_ids, attention_mask=src_attention_mask,
                    decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask,
                    labels=labels,
                    prior_input_ids=prior_input_ids, prior_attention_mask=prior_attn_mask,
                    posterior_input_ids=post_input_ids, posterior_attention_mask=post_attn_mask,
                    premise_cond_input_ids=premise_cond_input_ids,
                    premise_cond_attn_mask=premise_cond_attn_mask,
                    event_node_ids=trunc_node_list,
                    event_label=event_node_label,
                    event_distance=event_node_distance,
                    head=head_ids,
                    tail=tail_ids,
                    relation=relation,
                    triple_label=triple_label,
                    gumbel_src_input_ids=gumbel_src_input_ids, gumbel_src_attn_mask=gumbel_src_attn_mask,
                    gumbel_trg_pos_input_ids=gumbel_trg_pos_input_ids, gumbel_trg_pos_attn_mask=gumbel_trg_pos_attn_mask,
                    gumbel_trg_neg_input_ids=gumbel_trg_neg_input_ids, gumbel_trg_neg_attn_mask=gumbel_trg_neg_attn_mask,
                    clas_pos_input_ids=clas_pos_input_ids,
                    clas_pos_attn_mask=clas_pos_attn_mask,
                    clas_neg_input_ids=clas_neg_input_ids,
                    clas_neg_attn_mask=clas_neg_attn_mask,
                    current_fb_mode=fb_mode, current_beta=beta_t,
                    gumbel_temperature=args.gumbel_temperature,
                    used_lambda_clas=args.used_lambda_clas,
                    lambda_reg_z=args.lambda_reg_z,
                    lambda_cx=args.lambda_cx,
                    evtkg_lambda=args.evtkg_lambda,
                )
                if use_latent:
                    loss_clas = 0
                    loss, rec_loss, evt_loss, kld_loss, x_loss = outputs[:5]

                    loss_regz = 0
                    rec_loss = rec_loss.detach().item()
                    kld_loss = kld_loss.detach().item()
                    x_loss = x_loss.detach().item()
                    evt_loss = evt_loss.detach().item()

                    if args.used_lambda_clas > 0:
                        if epochid >=start_cla_epoch:
                            loss_clas = outputs[5]
                            loss_clas = loss_clas.detach().item()

                else:
                    loss = outputs[0]
                    rec_loss = 0
                    kld_loss = 0
                    x_loss = 0
                    loss_regz = 0
                    evt_loss=0
                    loss_clas=0
                accelerator.backward(loss)
                # checks whether the gradients are currently being synced across all processes.
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                # optimizer.zero_grad()
                if args.verbose:
                    progress_bar.update(1)
                tr_loss += loss.detach().item()
                global_step += 1

            if global_step % args.logging_steps == 0:
                logger.info(
                    "Step: {}, beta: {:.5f} | Loss: {:.5f}, rec-loss: {:.5f}, evt-loss: {:.10f}, kld-loss: {:.10f}, x_loss: {:.8f}, clas_loss: {:.8f}".format(
                        global_step, beta_t, (tr_loss - logging_loss) / args.logging_steps,
                        rec_loss, evt_loss, kld_loss, x_loss, loss_clas

                    ),
                )

                logging_loss = tr_loss

            # if global_step % args.validate_steps == 0:
            #     # if accelerator.is_main_process:
            #
            #     src_text = tokenizer.batch_decode(src_input_ids.detach().cpu().tolist())
            #
            #     gene_text = tokenizer.batch_decode(greedy_ids.detach().cpu().tolist())
            #     logger.info('============DEBUG==============')
            #     for _idx in range(len(src_text)):
            #         logger.info(src_text[_idx], gene_text[_idx])
            #     logger.info('============DEBUG==============')

    best_acc = 0
    args.used_lambda_clas = args.lambda_clas
    for epoch in range(args.num_train_epochs):
        if epoch >= start_cla_epoch:
            args.used_lambda_clas = args.lambda_clas
        else:
            args.used_lambda_clas = 0
        start_time = time.time()
        _train_epoch(epoch)
        accelerator.wait_for_everyone()
        if epoch < 5:
            continue
        # if accelerator.is_main_process:

        logger.info('=============Evaluation=============')
        # unwrapped_model = accelerator.unwrap_model(model)
        eval_start_time = time.time()
        dev_metrics, _dev_samples = _generation_epoch_for_evaluation(
            args, model, ent_model, tokenizer, dev_dataset, accelerator.device,
            split='dev', epoch=epoch
            )
        logger.info('dev metrics')
        logger.info(dev_metrics)

        test_metrics, _test_samples = _generation_epoch_for_evaluation(
            args, model, ent_model, tokenizer, test_dataset, accelerator.device,
            split='test', epoch=epoch

        )
        logger.info('test metrics')
        logger.info(test_metrics)
        eval_end_time = time.time()
        logger.info('evaluation use {:.4f} mins'.format((eval_end_time - eval_start_time) / 60))

        cur_acc = dev_metrics['entail_score']
        if cur_acc > best_acc:
            logger.info('best performance achieved at epoch {}, {}'.format(epoch, test_metrics))
        if accelerator.is_main_process:
            output_path = os.path.join(args.output_dir, f'checkpoint-epoch{epoch}.pt')

            unwrapped_model = accelerator.unwrap_model(model)

            torch.save(unwrapped_model.state_dict(), output_path)
            logger.info('model saved to {}'.format(output_path))
        best_acc = max(cur_acc, best_acc)

        end_time = time.time()
        logger.info('epoch {} use time {:.4f} mins'.format(epoch, (end_time - start_time) / 60))
        logger.info('*' * 20)




def main():
    args = get_config()
    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    args.device = torch.device("cuda:{}".format(args.gpu_id))
    set_seed(args.seed)
    # Setup logging
    set_logger(os.path.join(args.output_dir, "log.txt"))
    # accelerator.gradient_accumulation_steps = args.gradient_accumulation_steps
    # logger.info('set gradient_accumulation_steps to {}'.format(accelerator.gradient_accumulation_steps))
    # tokenizer_class, model_class = MODEL_CLASSES['bart']
    if args.do_train:
        checkout_point_path = args.model_name_or_path
    else:
        checkout_point_path = args.output_dir

    # classifier = BartForStoryEntailment.from_pretrained(args.classifier_path, num_classes=2)

    tokenizer = BartTokenizer.from_pretrained(checkout_point_path)
    model = BartVaeSeq2Seq(
        classifier_path=args.classifier_path, share_parameter=False,
        latent_embed_src=args.latent_embed_src==1, latent_embed_trg=args.latent_embed_trg==1,
        latent_memory=args.latent_memory==1,
        hop_num = args.hop_num, gamma = args.gamma, topk = args.topk,
        evtneg_weight=args.evtneg_weight,
        latent_size=args.latent_size, pad_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id, dim_target_kl=args.dim_target_kl

    )
    # model = BartForConditionalGeneration.from_pretrained(checkout_point_path)
    dev_samples = load_json_samples(args.dev_data_file, add_idx=True)
    test_samples = load_json_samples(args.test_data_file, add_idx=True)
    dev_dataset = TTEvalDataset(
        tokenizer, samples=dev_samples, source_length=args.source_length,
        target_length=args.target_length,
        max_cptlen=args.max_cptlen, max_trilen=args.max_trilen, max_evtlen=10,
        is_train=False
    )

    test_dataset = TTEvalDataset(
        tokenizer, samples=test_samples, source_length=args.source_length,
        target_length=args.target_length,
        max_cptlen=args.max_cptlen, max_trilen=args.max_trilen, max_evtlen=10,
        is_train=False
    )

    if args.do_train:
        train_samples = load_json_samples(args.train_data_file, add_idx=True)#[:4000]

        train(args, model, tokenizer, train_samples, dev_dataset, test_dataset)


if __name__ == '__main__':
    main()
