
import torch
import torch.utils.checkpoint
from torch import nn

from transformers.models.bart.configuration_bart import BartConfig

import logging
import copy
import numpy as np
from modeling_bart_vae import KGBartVaeForConditionalGeneration, BartCustomizedEncoder
from model_classifier import BartForStoryEntailment
logger = logging.getLogger()



def count_num_parameters(model):
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])



def init_para_from_bart_pretrained(m, pm, share_para=False):
    m.embed_tokens.weight = pm.embed_tokens.weight
    m.embed_positions.weight = pm.embed_positions.weight

    def share_parameter(val, share=True):
        return val if share else copy.deepcopy(val)

    for i in range(min(len(m.layers), len(pm.layers))):
        m.layers[i].self_attn.k_proj.weight = share_parameter(pm.layers[i].self_attn.k_proj.weight, share_para)
        m.layers[i].self_attn.k_proj.bias = share_parameter(pm.layers[i].self_attn.k_proj.bias, share_para)

        m.layers[i].self_attn.v_proj.weight = share_parameter(pm.layers[i].self_attn.v_proj.weight, share_para)
        m.layers[i].self_attn.v_proj.bias = share_parameter(pm.layers[i].self_attn.v_proj.bias, share_para)

        m.layers[i].self_attn.q_proj.weight = share_parameter(pm.layers[i].self_attn.q_proj.weight, share_para)
        m.layers[i].self_attn.q_proj.bias = share_parameter(pm.layers[i].self_attn.q_proj.bias, share_para)

        m.layers[i].self_attn.out_proj.weight = share_parameter(pm.layers[i].self_attn.out_proj.weight, share_para)
        m.layers[i].self_attn.out_proj.bias = share_parameter(pm.layers[i].self_attn.out_proj.bias, share_para)

        m.layers[i].self_attn_layer_norm.weight = share_parameter(pm.layers[i].self_attn_layer_norm.weight, share_para)
        m.layers[i].self_attn_layer_norm.bias = share_parameter(pm.layers[i].self_attn_layer_norm.bias, share_para)

        m.layers[i].fc1.weight = share_parameter(pm.layers[i].fc1.weight, share_para)
        m.layers[i].fc1.bias = share_parameter(pm.layers[i].fc1.bias, share_para)

        m.layers[i].fc2.weight = share_parameter(pm.layers[i].fc2.weight, share_para)
        m.layers[i].fc2.bias = share_parameter(pm.layers[i].fc2.bias, share_para)

        m.layers[i].final_layer_norm.weight = share_parameter(pm.layers[i].final_layer_norm.weight, share_para)
        m.layers[i].final_layer_norm.bias = share_parameter(pm.layers[i].final_layer_norm.bias, share_para)

    m.layernorm_embedding.weight = pm.layernorm_embedding.weight #if share_para else copy.deepcopy(pm.layernorm_embedding.weight)
    m.layernorm_embedding.bias = pm.layernorm_embedding.bias #if share_para else copy.deepcopy(pm.layernorm_embedding.bias)




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


def gaussian_kld_standard_prior(mu, logvar):
    """

    mu: latent mean 均值
    logvar: latent log variance 方差的对数
    """
    # KLD = -0.5 * torch.sum(logvar + 1 - mu.pow(2) - logvar.exp())
    KLD = -0.5 * (logvar + 1 - mu.pow(2) - logvar.exp())

    return KLD


def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * (1 + (recog_logvar - prior_logvar)
          - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
          - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)))
    return kld



def _normalize_z(z):
    assert z.dim() == 2
    z_min = torch.min(z, -1, keepdim=True).values.detach()
    z_ = z - z_min
    z_max = torch.max(z_, -1, keepdim=True).values.detach()
    return z_ / (z_max + 1e-10)


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature, use_gumbel=True):
    if use_gumbel:
        y = logits + sample_gumbel(logits.size()).to(logits.device)
    else:
        y = logits
    # return torch.softmax(y / temperature, dim=-1)  # TODO(hzt)
    return torch.softmax(y / temperature, dim=-1), torch.softmax(y / 0.00001, dim=-1)

def gumbel_softmax(logits, temperature, hard=False, use_gumbel=True):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y, yy = gumbel_softmax_sample(logits, temperature, use_gumbel=use_gumbel)

    if hard:  # TODO(hzt)
        y_argmax = torch.argmax(y, dim=-1)
        y_hard = torch.nn.functional.one_hot(y_argmax, num_classes=logits.shape[-1])
        y = y_hard - y.detach() + y
    # return y, yy  # TODO(hzt)
    # y = torch.nn.functional.gumbel_softmax(logits, tau=temperature, hard=hard)
    return y


class BartVaeSeq2Seq(nn.Module):
    def __init__(self, classifier_path, share_parameter=False,
                 latent_embed_src=False, latent_embed_trg=False, latent_memory=False,
                 hop_num=2, gamma=0.5, topk=4, evtneg_weight=0.5,
                 latent_size=128, pad_id=1,eos_token_id=2, bos_token_id=0, dim_target_kl=0.5):

        super().__init__()
        model_path = 'facebook/bart-base'
        self.transformer = KGBartVaeForConditionalGeneration.from_pretrained(
            model_path,
            latent_embed_src=latent_embed_src,
            latent_embed_trg=latent_embed_trg,
            latent_memory=latent_memory,
            hop_num=hop_num, gamma=gamma, topk=topk,
            evtneg_weight=evtneg_weight,
            pad_id=pad_id,
        )
        self.use_latent = latent_embed_src or latent_embed_trg or latent_memory
        self.latent_encoder = None
        self.latent_network = None
        self.z_linear = None
        self.classifier = None
        self.bilinear = None
        if self.use_latent:
            # latent encoder
            bart_config = BartConfig.from_pretrained(model_path)
            bart_config.customized_encoder_layers = 6
            latent_encoder = BartCustomizedEncoder(bart_config)
            init_para_from_bart_pretrained(
                latent_encoder, self.transformer.model.encoder, share_para=share_parameter)
            self.latent_encoder = latent_encoder
            self.latent_network = nn.Linear(bart_config.d_model, latent_size * 2)
            self.latent_z_linear = nn.Linear(latent_size, bart_config.d_model)

            pre_cond_encoder = BartCustomizedEncoder(bart_config)
            init_para_from_bart_pretrained(
                pre_cond_encoder, self.transformer.model.encoder, share_para=share_parameter)
            self.pre_cond_encoder = pre_cond_encoder
            self.contrast_precond_linear = nn.Linear(bart_config.d_model + latent_size, bart_config.d_model)
            # self.contrast_cond_linear = nn.Linear(bart_config.d_model + latent_size, 1)
            self.classifier = BartForStoryEntailment.from_pretrained(classifier_path, num_classes=2)
            # self.bilinear = nn.Linear(bart_config.d_model, bart_config.d_model)
        self.eos_id = eos_token_id
        self.bos_id = bos_token_id
        self.dim_target_kl = dim_target_kl

    def set_classifier_eval(self):
        for p in self.classifier.parameters():
            p.requires_grad = False
        self.classifier.eval()

    def sample_latent(self, encoder_hidden_states, attention_mask, num_samples=1):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """
        weight = attention_mask / attention_mask.sum(1, keepdim=True)
        pooled_out_final = torch.sum(encoder_hidden_states * weight.unsqueeze(-1), dim=1)

        # (batch_size, nz)
        mean, logvar = self.latent_network(pooled_out_final).chunk(2, -1)
        z = reparameterize(mean, logvar, num_samples)
        return z, mean, logvar, pooled_out_final


    def sample_latent_deterministic(self, encoder_hidden_states, attention_mask, num_samples=1):
        """
        Returns: Tensor1, Tensor2
            Tensor1: the tensor latent z with shape [batch, nsamples, nz]
            Tensor2: the tenor of KL for each x with shape [batch]
        """
        weight = attention_mask / attention_mask.sum(1, keepdim=True)
        pooled_out_final = torch.sum(encoder_hidden_states * weight.unsqueeze(-1), dim=1)

        # (batch_size, nz)
        mean, logvar = self.latent_network(pooled_out_final).chunk(2, -1)
        logvar.fill_(.0)
        z = mean
        # z = reparameterize(mean, logvar, num_samples)
        z = z.unsqueeze(1)
        return z, mean, logvar, pooled_out_final

    def sample_latent_z(self, input_ids, attention_mask):

        _encoder_outputs = self.latent_encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )
        encoder_hidden_states = _encoder_outputs[0]
        _z, _mu, _logvar, _ = self.sample_latent(
            encoder_hidden_states, attention_mask)
        latent_z = self.latent_z_linear(_z)
        return latent_z, _z, _mu, _logvar


    def gumbel_sequence_sample(self, src_input_ids, src_attention_mask, latent_z,
                               temperature, max_length, use_gumbel=True, hard=False):

        '''
        Returns:
            logits: the source logits of each token [B x seq_len x vsize]
            embeds: the representations of each token [B x  seq_len x hidden_dim]
        '''
        batch_size = src_input_ids.size(0)

        input_ids = torch.tensor([[self.eos_id, self.bos_id] for _ in range(batch_size)], device=src_input_ids.device)
        sample_mask = torch.ones(batch_size, max_length + 1, device=src_input_ids.device).type_as(input_ids)
        gumbel_weights = []
        logits = []
        greedy_tokens = []
        cur_len = 0
        input_emb = self.transformer.model.shared(input_ids)
        while cur_len <= max_length:
            outputs = self.transformer(
                input_ids=src_input_ids,
                attention_mask=src_attention_mask,
                decoder_inputs_embeds=input_emb,
                latent_z=latent_z,
                return_dict=False
            )

            next_token_logits = outputs[0][:, -1, :]
            pred_ids = torch.argmax(next_token_logits, dim=-1)
            greedy_tokens.append(pred_ids.unsqueeze(-1))

            g_weights = gumbel_softmax(next_token_logits, temperature, hard=hard, use_gumbel=use_gumbel)
            input_emb_ = torch.matmul(g_weights, self.transformer.model.shared.weight)  # TODO(hzt)

            input_emb_ = input_emb_.unsqueeze(1)
            input_emb = input_emb_ if input_emb is None else torch.cat((input_emb, input_emb_), dim=1)

            # if the input_emb is <|endoftext|>
            eos_probs = g_weights[:, self.eos_id].detach()
            not_eos = (eos_probs < 0.5).type_as(sample_mask)
            # sample_mask[:,cur_len+1:] = sample_mask[:, cur_len+1:] * not_eos.unsqueeze(-1)  # TODO(hzt)
            sample_mask[:, cur_len:] = sample_mask[:, cur_len:] * not_eos.unsqueeze(-1)

            gumbel_weights.append(g_weights)
            logits.append(next_token_logits)
            cur_len += 1

        logits = torch.stack(logits, 1)
        gumbel_weights = torch.stack(gumbel_weights, 1)
        generated_ids = torch.cat(greedy_tokens, -1)
        assert logits.size(1) == max_length + 1
        return logits, gumbel_weights, sample_mask, generated_ids

    def gumbel_sampling_with_fake_input(
            self, latent_z, gumbel_src_input_ids, gumbel_src_attn_mask,
            gumbel_trg_input_ids, gumbel_trg_attn_mask, gumbel_temperature=0.7):
        fn = self.transformer.model.decoder

        pos_gumbel_outputs = self.transformer(
            input_ids=gumbel_src_input_ids, attention_mask=gumbel_src_attn_mask,
            decoder_input_ids=gumbel_trg_input_ids,
            # decoder_inputs_embeds=gumbel_trg_input_ids,
            decoder_attention_mask=gumbel_trg_attn_mask,
            return_dict=False,
            latent_z=latent_z,

        )
        gumbel_logits = pos_gumbel_outputs[0]
        # [bs, s, v]
        gumbel_weights = gumbel_softmax(gumbel_logits, gumbel_temperature, hard=False,
                                            use_gumbel=True)


        inputs_embeds = torch.matmul(gumbel_weights, self.classifier.model.encoder.embed_tokens.weight)
        # inputs_embeds = inputs_embeds * gumbel_trg_attn_mask.unsqueeze(-1)
        inputs_embeds = inputs_embeds*fn.embed_scale
        return inputs_embeds


    def l1_norm_penalty(self, input_embeds, input_mask, target_ids, target_mask):
        fn = self.transformer.model.decoder
        xprime_inputs_embeds = fn.embed_tokens(target_ids) * fn.embed_scale
        mask_matrix = torch.bmm(
            input_mask.float().unsqueeze(-1),target_mask.float().unsqueeze(1))
        inner_product = torch.bmm(
            self.bilinear(input_embeds),
            xprime_inputs_embeds.transpose(1, 2)
        )
        l1_norm = torch.abs(inner_product * mask_matrix).sum() / mask_matrix.sum().clamp(min=1.0)
        return l1_norm

    def cal_contrastive_loss(self, precond_input_ids, precond_attention_mask, latent_z, prior_pooled):
        outputs = self.pre_cond_encoder(input_ids=precond_input_ids, attention_mask=precond_attention_mask,)
        encoder_hidden_states = outputs[0]
        weight = precond_attention_mask / precond_attention_mask.sum(1, keepdim=True)
        precond_pooled_out = torch.sum(encoder_hidden_states * weight.unsqueeze(-1), dim=1)
        # [bs, hs+d]
        observed = torch.cat((prior_pooled, latent_z.squeeze(1)), dim=-1)
        observed = self.contrast_precond_linear(observed)
        act = nn.Tanh()
        scores = torch.matmul(precond_pooled_out, act(observed).transpose(1,0))
        bs = precond_input_ids.size(0)
        labels = torch.eye(bs, device=precond_input_ids.device)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
        _loss = loss_fn(scores.view(-1), target=labels.view(-1))
        return _loss

    def forward(self, input_ids, attention_mask, decoder_input_ids,
                decoder_attention_mask, labels,
                prior_input_ids, prior_attention_mask,
                posterior_input_ids, posterior_attention_mask,
                premise_cond_input_ids,
                premise_cond_attn_mask,
                event_node_ids=None,
                event_label=None,
                event_distance=None,
                head=None,
                tail=None,
                relation=None,
                triple_label=None,
                # premise_input_ids, premise_attn_mask,
                # cond_input_ids, cond_attn_mask,
                gumbel_src_input_ids=None, gumbel_src_attn_mask=None,
                gumbel_trg_pos_input_ids=None, gumbel_trg_pos_attn_mask=None,
                gumbel_trg_neg_input_ids=None, gumbel_trg_neg_attn_mask=None,
                clas_pos_input_ids=None, clas_pos_attn_mask=None,
                clas_neg_input_ids=None, clas_neg_attn_mask=None,
                current_fb_mode=1, current_beta=1, gumbel_temperature=0.7,
                used_lambda_clas=1, lambda_reg_z=0.5,
                lambda_cx=0.1, evtkg_lambda=0.5
                ):
        """

        :param input_ids: [x,y,x']
        :param attention_mask:
        :param decoder_input_ids: [y']
        :param decoder_attention_mask:
        :param labels: [y']
        :param prior_input_ids: [x,y]
        :param prior_attention_mask:
        :param posterior_input_ids: [x',y']
        :param posterior_attention_mask:
        :param gumbel_src_input_ids: [x',y',x]
        :param gumbel_src_attn_mask:
        :param gumbel_trg_pos_input_ids: [y]
        :param gumbel_trg_pos_attn_mask:
        :param gumbel_trg_neg_input_ids: [y']
        :param gumbel_trg_neg_attn_mask:
        :param clas_pos_input_ids: [x]
        :param clas_pos_attn_mask:
        :param clas_neg_input_ids: [x']
        :param clas_neg_attn_mask:
        :return:
        """
        if not self.use_latent:
            outputs = self.transformer(
                input_ids=input_ids, attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask, labels=labels,
                event_node_ids=event_node_ids,
                event_label=event_label,
                event_distance=event_distance,
                head=head,
                tail=tail,
                relation=relation,
                triple_label=triple_label,
                latent_z=None,
                return_dict=False
            )
            return outputs
        # print(used_lambda_clas, current_fb_mode, current_beta)
        prior_encoder_outputs = self.latent_encoder(
            input_ids=prior_input_ids, attention_mask=prior_attention_mask, return_dict=False
        )
        prior_encoder_hidden_states = prior_encoder_outputs[0]
        if current_fb_mode == 2:
            prior_z, prior_mu, prior_logvar, prior_pooled = self.sample_latent_deterministic(
                prior_encoder_hidden_states, prior_attention_mask)

        else:
            prior_z, prior_mu, prior_logvar, prior_pooled = self.sample_latent(
                prior_encoder_hidden_states, prior_attention_mask)

        post_encoder_outputs = self.latent_encoder(
            input_ids=posterior_input_ids,
            attention_mask=posterior_attention_mask, return_dict=False
        )
        post_hidden_states = post_encoder_outputs[0]
        # [x,y]->z
        # [x',y']->z'
        # kl(z'||z)
        if current_fb_mode == 0:
            post_z, post_mu, post_logvar, post_pooled = self.sample_latent(
                post_hidden_states, posterior_attention_mask)
            # kld_loss = gaussian_kld_standard_prior(post_mu, post_logvar)
            kld_loss = gaussian_kld(post_mu, post_logvar, prior_mu, prior_logvar)
            loss_kl = kld_loss.sum(1)
        elif current_fb_mode == 1:
            post_z, post_mu, post_logvar, post_pooled = self.sample_latent(
                post_hidden_states, posterior_attention_mask)
            # kld_loss = gaussian_kld_standard_prior(post_mu, post_logvar)
            kld_loss = gaussian_kld(post_mu, post_logvar, prior_mu, prior_logvar)
            kl_mask = (kld_loss > self.dim_target_kl).float()
            loss_kl = (kl_mask * kld_loss).sum(dim=1)

        elif current_fb_mode == 2:
            post_z, post_mu, post_logvar, post_pooled = self.sample_latent_deterministic(
                post_hidden_states, posterior_attention_mask)
            # kld_loss = gaussian_kld_standard_prior(post_mu, post_logvar)
            kld_loss = gaussian_kld(post_mu, post_logvar, prior_mu, prior_logvar)
            loss_kl = kld_loss.sum(1)
        else:
            raise NotImplementedError
        loss_kl = loss_kl.mean()

        # latent_z = post_z
        # latent_z = self.latent_z_linear(post_z)
        # [x,y,x']->y'
        outputs = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask, labels=labels,
            return_dict=False,
            event_node_ids=event_node_ids,
            event_label=event_label,
            event_distance=event_distance,
            head=head,
            tail=tail,
            relation=relation,
            triple_label=triple_label,
            latent_z=self.latent_z_linear(post_z),

        )
        rec_loss = outputs[0]
        evt_pred_loss = outputs[1]
        # masked_lm_loss = masked_lm_loss + self.evtkg_lambda * evt_pred_loss

        # s=(c,x,y), pre_cond=(c,x')
        x_loss = self.cal_contrastive_loss(
            precond_input_ids=premise_cond_input_ids,
            precond_attention_mask=premise_cond_attn_mask,
            latent_z=post_z, prior_pooled=prior_pooled)



        loss_clas = 0
        if used_lambda_clas > 0:
            # [x',y'x,z']->hat(y)

            y_hat_inputs_embeds = self.gumbel_sampling_with_fake_input(
                latent_z=self.latent_z_linear(post_z),
                gumbel_src_input_ids=gumbel_src_input_ids, gumbel_src_attn_mask=gumbel_src_attn_mask,
                gumbel_trg_input_ids=gumbel_trg_pos_input_ids, gumbel_trg_attn_mask=gumbel_trg_pos_attn_mask,
                gumbel_temperature=gumbel_temperature
            )

            pos_labels = torch.ones(
                clas_pos_input_ids.size(0), device=clas_pos_input_ids.device, dtype=torch.long)
            neg_labels = torch.zeros(
                clas_pos_input_ids.size(0), device=clas_pos_input_ids.device, dtype=torch.long)

            # [x,hat(y)]->1
            outputs1 = self.classifier(
                prefix_input_ids=clas_pos_input_ids,
                prefix_attention_mask=clas_pos_attn_mask,
                suffix_inputs_embeds=y_hat_inputs_embeds[:, 1:, :],
                suffix_attention_mask=gumbel_trg_pos_attn_mask[:, 1:],
                labels=pos_labels
            )
            pos_loss1 = outputs1[0]

            # [x',hat(y)]->0
            outputs4 = self.classifier(
                prefix_input_ids=clas_neg_input_ids,
                prefix_attention_mask=clas_neg_attn_mask,
                suffix_inputs_embeds=y_hat_inputs_embeds[:, 1:, :],
                suffix_attention_mask=gumbel_trg_pos_attn_mask[:, 1:],
                labels=neg_labels
            )
            neg_loss2 = outputs4[0]
            loss_clas = (pos_loss1+neg_loss2)/2


        loss = rec_loss + evtkg_lambda*evt_pred_loss + current_beta * loss_kl + lambda_cx * x_loss

        if used_lambda_clas >0:
            loss = loss+used_lambda_clas*loss_clas
            return tuple((loss, rec_loss, evt_pred_loss, loss_kl, x_loss, loss_clas) + outputs[2:])

        return tuple((loss, rec_loss, evt_pred_loss, loss_kl, x_loss) + outputs[2:])




