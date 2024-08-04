import torch
from torch.utils.data import Dataset


class TTEvalDataset(Dataset):

    def __init__(self, tokenizer, samples, source_length, target_length,
                 max_cptlen=240, max_trilen=300, max_evtlen=10, is_train=False):

        self.tokenizer = tokenizer
        self.source_length = source_length
        self.target_length = target_length
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id
        self.sep_id = tokenizer.sep_token_id
        self.samples = samples
        self.src_input_ids = []
        self.trg_input_ids = []
        self.sample_idx = []
        self.posterior_input_ids = []
        self.prior_input_ids = []
        self.gumbel_input_ids = []
        self.classifier_pos_input_ids = []
        self.classifier_neg_input_ids = []
        self.event_node_list, self.event_node_label, self.event_node_distance = [], [], []
        self.head_ids, self.tail_ids, self.relation, self.triple_label = [], [], [], []
        max_evtnum, max_evttri_num = 0, 0

        for sample_idx, d in enumerate(self.samples):

            premise = d['premise']
            initial = d['initial']
            counterfactual = d['counterfactual']
            original_ending = d['original_ending']
            if is_train:
                edited_ending = d['edited_ending']
            else:
                edited_ending = d['edited_endings'][0]
            edited_ending = ' '.join(edited_ending)
            original_story = ' '.join((premise, initial, original_ending))
            original_story_ids = tokenizer.encode(' ' + original_story, add_special_tokens=False)
            cf_ctx_ids = tokenizer.encode(' ' + premise + ' ' + counterfactual, add_special_tokens=False)
            edited_ending_ids = tokenizer.encode(' ' + edited_ending, add_special_tokens=False)

            src_ids = [self.bos_id] + original_story_ids + [self.sep_id] + cf_ctx_ids + [self.eos_id]
            trg_ids = [self.bos_id] + edited_ending_ids

            self.src_input_ids.append(src_ids)
            self.trg_input_ids.append(trg_ids)
            self.sample_idx.append(sample_idx)
            prior_ids = [self.bos_id] + original_story_ids + [self.eos_id]
            post_ids = [self.bos_id] + cf_ctx_ids + edited_ending_ids + [self.eos_id]
            self.prior_input_ids.append(prior_ids)
            self.posterior_input_ids.append(post_ids)

            self.event_node_list.append(d['evt_ids_list'])
            self.event_node_label.append(d['evt_labels'])
            self.event_node_distance.append(d['evt_distance'])
            self.head_ids.append(d['evt_head_ids'])
            self.tail_ids.append(d['evt_tail_ids'])
            self.relation.append(d['evt_tri_rels'])
            self.triple_label.append(d['evt_tri_labels'])

            max_evtnum = max(max_evtnum, len(d['evt_labels']))
            max_evttri_num = max(max_evttri_num, len(d['evt_head_ids']))

        self.max_trilen, self.max_evtlen = max_evttri_num, 10
        self.max_cptlen = max_evtnum



    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        src_input_ids = [idx for idx in self.src_input_ids[i]]
        src_attention_mask = [1] * len(src_input_ids)
        while len(src_input_ids) < self.source_length:
            src_input_ids.append(self.pad_id)
            src_attention_mask.append(0)

        prior_input_ids = [idx for idx in self.prior_input_ids[i]]
        prior_attn_mask = [1] * len(prior_input_ids)
        while len(prior_input_ids) < 128:
            prior_input_ids.append(self.pad_id)
            prior_attn_mask.append(0)

        post_input_ids = [idx for idx in self.posterior_input_ids[i]]
        post_attn_mask = [1] * len(post_input_ids)
        while len(post_input_ids) < 128:
            post_input_ids.append(self.pad_id)
            post_attn_mask.append(0)

        trg_inputs = self.trg_input_ids[i]
        decoder_input_ids = [self.eos_id] + trg_inputs
        decoder_attention_mask = [1] * len(decoder_input_ids)
        labels = trg_inputs + [self.eos_id]
        while len(decoder_input_ids) < self.target_length:
            decoder_input_ids.append(self.pad_id)
            decoder_attention_mask.append(0)
            labels.append(-100)

        event_node_label = self.event_node_label[i]
        event_node_distance = self.event_node_distance[i]
        trunc_node_list = []
        for node in self.event_node_list[i]:
            if len(node) > self.max_evtlen:
                node = node[:self.max_evtlen]
            else:
                node = node + [self.pad_id] * (self.max_evtlen - len(node))
            trunc_node_list.append(node)
        if len(trunc_node_list) > self.max_cptlen:
            trunc_node_list = trunc_node_list[:self.max_cptlen]
            event_node_distance = event_node_distance[:self.max_cptlen]
            event_node_label = event_node_label[:self.max_cptlen]

        pad_evt = [self.pad_id] * self.max_evtlen
        while len(trunc_node_list) < self.max_cptlen:
            trunc_node_list.append(pad_evt)
            event_node_label.append(-1)
            event_node_distance.append(0)

        head_ids = self.head_ids[i]
        tail_ids = self.tail_ids[i]
        relation = self.relation[i]
        triple_label = self.triple_label[i]
        if len(head_ids) > self.max_trilen:
            head_ids = head_ids[:self.max_trilen]
            tail_ids = tail_ids[:self.max_trilen]
            relation = relation[:self.max_trilen]
            triple_label = triple_label[:self.max_trilen]

        while len(head_ids) < self.max_trilen:
            head_ids.append(0)
            tail_ids.append(0)
            relation.append(0)
            triple_label.append(-1)

        res = [src_input_ids, src_attention_mask,
               decoder_input_ids, decoder_attention_mask, labels,
               prior_input_ids, prior_attn_mask,
               post_input_ids, post_attn_mask,
               trunc_node_list, event_node_label, event_node_distance,
               head_ids, tail_ids, relation, triple_label,
               self.sample_idx[i],

               ]

        res = [torch.tensor(r) for r in res]
        return res


class RocStoriesDataset(Dataset):

    def __init__(self, tokenizer, samples, source_length, target_length,
                 max_cptlen=240, max_trilen=300, max_evtlen=10, is_train=False):

        self.tokenizer = tokenizer
        self.source_length = source_length
        self.target_length = target_length
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id
        self.sep_id = tokenizer.sep_token_id
        self.samples = samples
        self.src_input_ids = []
        self.trg_input_ids = []
        self.sample_idx = []
        self.posterior_input_ids = []
        self.prior_input_ids = []
        self.premise_cond_input_ids = []

        self.gumbel_src_input_ids = []
        self.gumbel_trg_pos_input_ids = []
        self.gumbel_trg_neg_input_ids = []

        self.classifier_pos_input_ids = []
        self.classifier_neg_input_ids = []
        self.event_node_list, self.event_node_label, self.event_node_distance = [], [], []
        self.head_ids, self.tail_ids, self.relation, self.triple_label = [], [], [], []
        max_evtnum, max_evttri_num = 0, 0

        for sample_idx, d in enumerate(self.samples):

            premise = d['premise']
            initial = d['initial']
            counterfactual = d['counterfactual']
            original_ending = d['original_ending']
            if is_train:
                edited_ending = d['edited_ending']
            else:
                edited_ending = d['edited_endings'][0]
            edited_ending = ' '.join(edited_ending)
            original_story = ' '.join((premise, initial, original_ending))
            original_story_ids = tokenizer.encode(' '+original_story, add_special_tokens=False)
            cf_ctx_ids = tokenizer.encode(' '+premise+' '+counterfactual, add_special_tokens=False)
            edited_ending_ids = tokenizer.encode(' '+edited_ending, add_special_tokens=False)

            src_ids = [self.bos_id]+original_story_ids+[self.sep_id]+cf_ctx_ids+[self.eos_id]
            trg_ids = [self.bos_id]+edited_ending_ids

            self.src_input_ids.append(src_ids)
            self.trg_input_ids.append(trg_ids)
            self.sample_idx.append(sample_idx)

            prior_ids = [self.bos_id]+original_story_ids+[self.eos_id]
            post_ids = [self.bos_id]+cf_ctx_ids+edited_ending_ids+[self.eos_id]
            self.prior_input_ids.append(prior_ids)
            self.posterior_input_ids.append(post_ids)

            premise_cond_ids = tokenizer.encode(f' {premise} {counterfactual}')
            self.premise_cond_input_ids.append(premise_cond_ids)

            ctx_ids = tokenizer.encode(f' {premise} {initial}', add_special_tokens=False)
            gumbel_src_input_ids = [self.bos_id] + cf_ctx_ids + edited_ending_ids +[self.sep_id]+ctx_ids+ [self.eos_id]
            self.gumbel_src_input_ids.append(gumbel_src_input_ids)

            original_ending_ids = tokenizer.encode(' '+original_ending, add_special_tokens=False)
            gumbel_trg_pos_input_ids = [self.bos_id]+original_ending_ids
            self.gumbel_trg_pos_input_ids.append(gumbel_trg_pos_input_ids)
            gumbel_trg_neg_input_ids = [self.bos_id]+edited_ending_ids
            self.gumbel_trg_neg_input_ids.append(gumbel_trg_neg_input_ids)

            self.classifier_pos_input_ids.append(ctx_ids)
            self.classifier_neg_input_ids.append(cf_ctx_ids)

            self.event_node_list.append(d['evt_ids_list'])
            self.event_node_label.append(d['evt_labels'])
            self.event_node_distance.append(d['evt_distance'])
            self.head_ids.append(d['evt_head_ids'])
            self.tail_ids.append(d['evt_tail_ids'])
            self.relation.append(d['evt_tri_rels'])
            self.triple_label.append(d['evt_tri_labels'])

            max_evtnum = max(max_evtnum, len(d['evt_labels']))
            max_evttri_num = max(max_evttri_num, len(d['evt_head_ids']))

        self.max_trilen, self.max_evtlen = max_evttri_num, 10
        self.max_cptlen = max_evtnum

    # self.max_trilen, self.max_evtlen = max_trilen, max_evtlen
    #     self.max_cptlen = max_cptlen

    def __len__(self):
        return len(self.samples)

    def process_encoder_inputs(self, input_ids, maxlen, pad_left=False):

        _input_ids = [idx for idx in input_ids]
        _attn_mask = [1] * len(_input_ids)
        if not pad_left:
            while len(_input_ids) < maxlen:
                _input_ids.append(self.pad_id)
                _attn_mask.append(0)
        else:
            _pad_size = maxlen - len(_input_ids)
            _input_ids = [self.pad_id] * _pad_size + _input_ids
            _attn_mask = [0] * _pad_size + _attn_mask
        return _input_ids, _attn_mask

    def __getitem__(self, i):

        src_input_ids, src_attention_mask = self.process_encoder_inputs(
            self.src_input_ids[i], self.source_length, pad_left=False
        )
        prior_input_ids, prior_attn_mask = self.process_encoder_inputs(
            self.prior_input_ids[i], 128, pad_left=False
        )

        post_input_ids, post_attn_mask = self.process_encoder_inputs(
            self.posterior_input_ids[i], 128, pad_left=False
        )

        trg_inputs = self.trg_input_ids[i]
        decoder_input_ids = [self.eos_id] + trg_inputs
        decoder_attention_mask = [1]*len(decoder_input_ids)
        labels = trg_inputs + [self.eos_id]
        while len(decoder_input_ids) < self.target_length:
            decoder_input_ids.append(self.pad_id)
            decoder_attention_mask.append(0)
            labels.append(-100)

        premise_cond_input_ids, premise_cond_attn_mask = self.process_encoder_inputs(
            self.premise_cond_input_ids[i], 96, pad_left=False
        )

        event_node_label = self.event_node_label[i]
        event_node_distance = self.event_node_distance[i]
        trunc_node_list = []
        for node in self.event_node_list[i]:
            if len(node) > self.max_evtlen:
                node = node[:self.max_evtlen]
            else:
                node = node + [self.pad_id] * (self.max_evtlen - len(node))
            trunc_node_list.append(node)
        if len(trunc_node_list) > self.max_cptlen:
            trunc_node_list = trunc_node_list[:self.max_cptlen]
            event_node_distance = event_node_distance[:self.max_cptlen]
            event_node_label = event_node_label[:self.max_cptlen]

        pad_evt = [self.pad_id] * self.max_evtlen
        while len(trunc_node_list) < self.max_cptlen:
            trunc_node_list.append(pad_evt)
            event_node_label.append(-1)
            event_node_distance.append(0)

        head_ids = self.head_ids[i]
        tail_ids = self.tail_ids[i]
        relation = self.relation[i]
        triple_label = self.triple_label[i]
        if len(head_ids) > self.max_trilen:
            head_ids = head_ids[:self.max_trilen]
            tail_ids = tail_ids[:self.max_trilen]
            relation = relation[:self.max_trilen]
            triple_label = triple_label[:self.max_trilen]

        while len(head_ids) < self.max_trilen:
            head_ids.append(0)
            tail_ids.append(0)
            relation.append(0)
            triple_label.append(-1)

        gumbel_src_input_ids, gumbel_src_attn_mask = self.process_encoder_inputs(
            self.gumbel_src_input_ids[i], self.source_length
        )

        gumbel_trg_pos_input_ids = [self.eos_id]+self.gumbel_trg_pos_input_ids[i]
        gumbel_trg_pos_attn_mask = [1]*len(gumbel_trg_pos_input_ids)
        trg_pad_size = self.target_length-len(gumbel_trg_pos_input_ids)
        gumbel_trg_pos_input_ids = gumbel_trg_pos_input_ids+[self.pad_id]*trg_pad_size
        gumbel_trg_pos_attn_mask = gumbel_trg_pos_attn_mask+[0]*trg_pad_size

        gumbel_trg_neg_input_ids = [self.eos_id]+self.gumbel_trg_neg_input_ids[i]
        gumbel_trg_neg_attn_mask = [1]*len(gumbel_trg_neg_input_ids)
        trg_pad_size = self.target_length-len(gumbel_trg_neg_attn_mask)
        gumbel_trg_neg_input_ids = gumbel_trg_neg_input_ids+[self.pad_id]*trg_pad_size
        gumbel_trg_neg_attn_mask = gumbel_trg_neg_attn_mask+[0]*trg_pad_size


        prefix_maxlen = 50
        suffix_maxlen = 80

        clas_pos_input_ids, clas_pos_attn_mask = self.process_encoder_inputs(
            [self.bos_id] + self.classifier_pos_input_ids[i], prefix_maxlen, pad_left=True
        )

        clas_neg_input_ids, clas_neg_attn_mask = self.process_encoder_inputs(
            [self.bos_id] + self.classifier_neg_input_ids[i], prefix_maxlen, pad_left=True
        )

        res = [src_input_ids, src_attention_mask,
               decoder_input_ids, decoder_attention_mask, labels,
               prior_input_ids, prior_attn_mask,
               post_input_ids, post_attn_mask,
               premise_cond_input_ids, premise_cond_attn_mask,

               trunc_node_list, event_node_label, event_node_distance,
               head_ids, tail_ids, relation, triple_label,

               gumbel_src_input_ids, gumbel_src_attn_mask,
               gumbel_trg_pos_input_ids, gumbel_trg_pos_attn_mask,
               gumbel_trg_neg_input_ids, gumbel_trg_neg_attn_mask,
               clas_pos_input_ids, clas_pos_attn_mask,
               clas_neg_input_ids, clas_neg_attn_mask,
               ]
        res = [torch.tensor(r) for r in res]
        return res

