#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from figet import Constants
from figet.hyperbolic import PoincareDistance
from . import utils
from figet.model_utils import CharEncoder, SelfAttentiveSum, sort_batch_by_length
from math import pi

log = utils.get_logging()


class MentionEncoder(nn.Module):

    def __init__(self, char_vocab, args):
        super(MentionEncoder, self).__init__()
        self.char_encoder = CharEncoder(char_vocab, args)
        self.attentive_weighted_average = SelfAttentiveSum(args.emb_size, 1)
        self.dropout = nn.Dropout(args.mention_dropout)

    def forward(self, mentions, mention_chars, word_lut):
        mention_embeds = word_lut(mentions)             # batch x mention_length x emb_size

        weighted_avg_mentions, _ = self.attentive_weighted_average(mention_embeds)
        char_embed = self.char_encoder(mention_chars)
        output = torch.cat((weighted_avg_mentions, char_embed), 1)
        return self.dropout(output)


class ContextEncoder(nn.Module):

    def __init__(self, args):
        self.emb_size = args.emb_size
        self.pos_emb_size = args.positional_emb_size
        self.rnn_size = args.context_rnn_size
        self.hidden_attention_size = 100
        super(ContextEncoder, self).__init__()
        self.pos_linear = nn.Linear(1, self.pos_emb_size)
        self.context_dropout = nn.Dropout(args.context_dropout)
        self.rnn = nn.LSTM(self.emb_size + self.pos_emb_size, self.rnn_size, bidirectional=True, batch_first=True)
        self.attention = SelfAttentiveSum(self.rnn_size * 2, self.hidden_attention_size) # x2 because of bidirectional

    def forward(self, contexts, positions, context_len, word_lut, hidden=None):
        """
        :param contexts: batch x max_seq_len
        :param positions: batch x max_seq_len
        :param context_len: batch x 1
        """
        positional_embeds = self.get_positional_embeddings(positions)   # batch x max_seq_len x pos_emb_size
        ctx_word_embeds = word_lut(contexts)                            # batch x max_seq_len x emb_size
        ctx_embeds = torch.cat((ctx_word_embeds, positional_embeds), 2)

        ctx_embeds = self.context_dropout(ctx_embeds)

        rnn_output = self.sorted_rnn(ctx_embeds, context_len)

        return self.attention(rnn_output)

    def get_positional_embeddings(self, positions):
        """ :param positions: batch x max_seq_len"""
        pos_embeds = self.pos_linear(positions.view(-1, 1))                     # batch * max_seq_len x pos_emb_size
        return pos_embeds.view(positions.size(0), positions.size(1), -1)        # batch x max_seq_len x pos_emb_size

    def sorted_rnn(self, ctx_embeds, context_len):
        sorted_inputs, sorted_sequence_lengths, restoration_indices = sort_batch_by_length(ctx_embeds, context_len)
        packed_sequence_input = pack(sorted_inputs, sorted_sequence_lengths, batch_first=True)
        packed_sequence_output, _ = self.rnn(packed_sequence_input, None)
        unpacked_sequence_tensor, _ = unpack(packed_sequence_output, batch_first=True)
        return unpacked_sequence_tensor.index_select(0, restoration_indices)


class Projector(nn.Module):

    def __init__(self, args, input_size):
        self.args = args
        super(Projector, self).__init__()
        self.W_in = nn.Linear(input_size, args.hidden_size, bias=args.bias == 1)
        self.hidden_layers = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size, bias=args.bias == 1)
                                            for _ in range(args.hidden_layers)])
        self.W_out = nn.Linear(args.hidden_size, args.type_dims, bias=args.bias == 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=args.projection_dropout)

        self.scaler = nn.Linear(input_size, 1, bias=args.bias == 1)
        self.sigmoid = nn.Sigmoid()

        for layer in [self.W_in, self.W_out] + [l for l in self.hidden_layers]:
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, input):
        direction_vectors = self.get_direction_vector(input)
        scalers = self.get_scalers(input)

        return direction_vectors * scalers

    def get_direction_vector(self, input):
        hidden_state = self.dropout(self.relu(self.W_in(input)))
        for layer in self.hidden_layers:
            hidden_state = self.dropout(self.relu(layer(hidden_state)))

        output = self.W_out(hidden_state)  # batch x type_dims

        norms = output.norm(p=2, dim=1, keepdim=True)
        return output.div(norms.expand_as(output))

    def get_scalers(self, input):
        output = self.scaler(input)
        return self.sigmoid(output)


class Model(nn.Module):

    def __init__(self, args, vocabs):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args = args
        type_vocab = vocabs[Constants.TYPE_VOCAB]
        self.coarse_ids = type_vocab.get_coarse_ids()
        self.fine_ids = type_vocab.get_fine_ids()
        self.ultrafine_ids = type_vocab.get_ultrafine_ids()
        self.ids = [self.coarse_ids, self.fine_ids, self.ultrafine_ids]

        super(Model, self).__init__()
        self.word_lut = nn.Embedding(vocabs[Constants.TOKEN_VOCAB].size_of_word2vecs(), args.emb_size,
                                     padding_idx=Constants.PAD)
        self.type_lut = nn.Embedding(vocabs[Constants.TYPE_VOCAB].size(), args.type_dims)

        self.mention_encoder = MentionEncoder(vocabs[Constants.CHAR_VOCAB], args)
        self.context_encoder = ContextEncoder(args)
        self.feature_len = args.context_rnn_size * 2 + args.emb_size + args.char_emb_size

        self.coarse_projector = Projector(args, self.feature_len)
        self.fine_projector = Projector(args, self.feature_len + args.type_dims)
        self.ultrafine_projector = Projector(args, self.feature_len + args.type_dims)

        self.hyperbolic = args.metric == "hyperbolic"
        self.distance_function = PoincareDistance.apply if self.hyperbolic else nn.PairwiseDistance()
        self.cos_sim_function = nn.CosineSimilarity()
        self.hinge_loss_function = nn.HingeEmbeddingLoss()

    def init_params(self, word2vec, type2vec):
        self.word_lut.weight.data.copy_(word2vec)
        self.word_lut.weight.requires_grad = False
        self.type_lut.weight.data.copy_(type2vec)
        self.type_lut.weight.requires_grad = False

    def forward(self, input):
        contexts, positions, context_len = input[0], input[1], input[2]
        mentions, mention_chars = input[3], input[4]
        type_indexes = input[5]

        mention_vec = self.mention_encoder(mentions, mention_chars, self.word_lut)
        context_vec, attn = self.context_encoder(contexts, positions, context_len, self.word_lut)

        input_vec = torch.cat((mention_vec, context_vec), dim=1)

        coarse_embed = self.coarse_projector(input_vec)

        fine_input = torch.cat((input_vec, coarse_embed), dim=1)
        fine_embed = self.fine_projector(fine_input)

        ultrafine_input = torch.cat((input_vec, fine_embed), dim=1)
        ultrafine_embed = self.ultrafine_projector(ultrafine_input)
        predictions = [coarse_embed, fine_embed, ultrafine_embed]

        final_loss = 0
        loss, avg_angle, dist_to_pos, euclid_dist = [], [], [], []
        if type_indexes is not None:
            for pred_embed_i, gran_ids in zip(predictions, self.ids):
                loss_i, avg_angle_i, dist_to_pos_i, euclid_dist_i = self.calculate_loss(pred_embed_i, type_indexes, gran_ids)
                loss.append(loss_i)
                avg_angle.append(avg_angle_i)
                dist_to_pos.append(dist_to_pos_i)
                euclid_dist.append(euclid_dist_i)

            final_loss = sum(loss)

        return final_loss, predictions, attn, avg_angle, dist_to_pos, euclid_dist

    def calculate_loss(self, predicted_embeds, type_indexes, granularity_ids):
        types_by_gran = self.filter_types_by_granularity(type_indexes, granularity_ids)

        type_lut_ids = [idx for row in types_by_gran for idx in row]
        index_on_prediction = self.get_index_on_prediction(types_by_gran)

        if len(type_lut_ids) == 0:
            return torch.zeros(1, requires_grad=True).to(self.device), torch.zeros(1).to(self.device), \
                   torch.zeros(1).to(self.device), torch.zeros(1).to(self.device)

        true_type_embeds = self.type_lut(torch.LongTensor(type_lut_ids).to(self.device))     # len_type_lut_ids x type_dims
        expanded_predicted = predicted_embeds[index_on_prediction]

        distances_to_pos = self.distance_function(expanded_predicted, true_type_embeds)
        metric_distance = distances_to_pos**2 if self.hyperbolic else distances_to_pos

        cosine_similarity = self.cos_sim_function(expanded_predicted, true_type_embeds)
        cosine_distance = 1 - cosine_similarity

        distance_sum = self.args.metric_dist_factor * metric_distance + self.args.cosine_dist_factor * cosine_distance

        y = torch.ones(len(distance_sum)).to(self.device)
        loss = self.hinge_loss_function(distance_sum, y)

        # stats
        avg_angle = torch.acos(torch.clamp(cosine_similarity.detach(), min=-1, max=1)) * 180 / pi
        euclid_dist = nn.PairwiseDistance()(expanded_predicted.detach(), true_type_embeds.detach())

        return loss, avg_angle, distances_to_pos.detach(), euclid_dist

    def filter_types_by_granularity(self, type_indexes, gran_ids):
        result = []
        for row in type_indexes:
            row_result = [idx for idx in row.tolist() if idx in gran_ids]
            result.append(row_result)
        return result

    def get_index_on_prediction(self, types_by_gran):
        indexes = []
        for index, type_id in enumerate(types_by_gran):
            for i in range(len(type_id)):
                indexes.append(index)
        return indexes
