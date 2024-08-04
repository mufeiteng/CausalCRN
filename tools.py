import os
import codecs
import numpy as np
import json
import logging
import random
import torch
import multiprocessing
import torch.nn as nn

userpath = os.path.expanduser('~')
all_data_path = os.path.join(userpath, 'Documents/sources')
# all_data_path = '/home/share/feiteng/Documents/sources'
project_data_path = os.path.join(all_data_path, 'CausalCRN')


def generate_batches(data, batch_size, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]


def load_json_samples(path, add_idx=False):
    _samples = []
    idx = 0
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            d = json.loads(line)
            if not add_idx:
                d['_sample_idx'] = idx
            _samples.append(d)
            idx += 1
    return _samples



def save_data(samples, outpath):
    with codecs.open(outpath, 'w', 'utf-8') as fout:
        for d in samples:
            fout.write(json.dumps(d))
            fout.write('\n')



def set_logger(logfile=None):
    console = logging.StreamHandler()
    handlers = [console]
    if logfile:
        file_handler = logging.FileHandler(logfile, "w")
        handlers.append(file_handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s: %(name)s: %(message)s',
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=handlers
    )

class JsonDumpHelper(json.JSONEncoder):
    def default(self, obj):
        if type(obj) != str:
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def multiprocessing_tokenize_data(implementor, data, tokenizer, *args, **kwargs):
    workers = multiprocessing.cpu_count() - 2
    pool = multiprocessing.Pool()
    filesize = len(data)
    _results = []
    for i in range(workers):
        chunk_start, chunk_end = (filesize * i) // workers, (filesize * (i + 1)) // workers
        _results.append(pool.apply_async(
            implementor, (data[chunk_start:chunk_end], tokenizer, args, kwargs,)
        ))
    pool.close()
    pool.join()
    _total = []
    for _result in _results:
        samples = _result.get()
        _total.extend(samples)
    assert len(_total) == filesize
    return _total


def reparameterize(mu, logvar, nsamples=1):
    """sample from posterior Gaussian family
    Args:
        mu: Tensor
            Mean of gaussian distribution with shape (batch, nz)
        logvar: Tensor
            logvar of gaussian distibution with shape (batch, nz)
    Returns: Tensor
        Sampled z with shape (batch, nsamples, nz)
        :param nsamples:
    """
    batch_size, nz = mu.size()
    std = logvar.mul(0.5).exp()
    mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
    std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)
    eps = torch.zeros_like(std_expd).normal_()
    return mu_expd + torch.mul(eps, std_expd)


def get_num_trainable_params(model):
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * (1 + (recog_logvar - prior_logvar)
          - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
          - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)))
    return kld


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


