# -*-coding:utf-8-*-
from CausalCRN.evaluation.bleu_custom import compute_bleu, sentence_bleu_scorer
from CausalCRN.evaluation.rouge_custom import corpus_rouge_moses
from CausalCRN.evaluation.distinct import distinct_n_corpus_level
from CausalCRN.evaluation.meteor_custom import corpus_meteor_moses
from nltk import word_tokenize
from bert_score import score as bert_score
import torch


def _clean_text(txt):
    return txt.lower()



def eval_bert_score(refs, predictions, bert_model="bert-base-uncased"):

    references = []
    hypotheses = []
    for i in range(len(refs)):
        gold_end = refs[i]
        predicted_ending = predictions[i]
        clean_reference = _clean_text(gold_end)
        clean_hypothesis = _clean_text(predicted_ending)
        if len(clean_hypothesis) == 0:
            continue
        references.append(clean_reference)
        hypotheses.append(clean_hypothesis)

    P, R, F1 = bert_score(hypotheses, references, model_type=bert_model,
                          device='cuda' if torch.cuda.is_available() else 'cpu')
    return {
        # "bert_score_P": P.mean().item(),
        # "bert_score_R": R.mean().item(),
        "bert_score_F1": F1.mean().item(),
        # "bert_score_P_by_instance": [float(f) for f in list(P.numpy())],
        # "bert_score_R_by_instance": [float(f) for f in list(R.numpy())],
        # "bert_score_F1_by_instance": [float(f) for f in list(F1.numpy())],
    }


def eval_from_json_samples(samples):
    gold_text, gened_text = [], []
    predictions, references = [], []
    for d in samples:
        source = d['source']
        target = d['target']
        generated = d['generated']
        predictions.append(word_tokenize(generated))
        references.append([word_tokenize(target)])
        gold_text.append(target)
        gened_text.append(generated)

    meteor = corpus_meteor_moses(list_of_refs=references, list_of_hypos=predictions)
    bleu2 = compute_bleu(reference_corpus=references, translation_corpus=predictions, max_order=2)[0]
    bleu4 = compute_bleu(reference_corpus=references, translation_corpus=predictions, max_order=4)[0]
    dis1 = distinct_n_corpus_level(predictions, 1)
    dis2 = distinct_n_corpus_level(predictions, 2)
    rouges = corpus_rouge_moses(list_of_hypos=predictions, list_of_refs=references)
    rougel = rouges[-1]

    berts = eval_bert_score(refs=gold_text, predictions=gened_text)
    metrics = {
        'bleu2': bleu2, 'bleu4': bleu4, 'dis1': dis1, 'dis2': dis2,
        'rougel': rougel, 'berts': berts['bert_score_F1']
    }
    res = dict()
    for k in metrics:
        res[k] = '{:.4f}'.format(metrics[k])
    return res
