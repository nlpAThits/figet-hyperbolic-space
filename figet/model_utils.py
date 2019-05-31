import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


def sort_batch_by_length(tensor: torch.autograd.Variable, sequence_lengths: torch.autograd.Variable):
    """
    @ from allennlp
    Sort a batch first tensor by some specified lengths.

    Parameters
    ----------
    tensor : Variable(torch.FloatTensor), required.
        A batch first Pytorch tensor.
    sequence_lengths : Variable(torch.LongTensor), required.
        A tensor representing the lengths of some dimension of the tensor which
        we want to sort by.

    Returns
    -------
    sorted_tensor : Variable(torch.FloatTensor)
        The original tensor sorted along the batch dimension with respect to sequence_lengths.
    sorted_sequence_lengths : Variable(torch.LongTensor)
        The original sequence_lengths sorted by decreasing size.
    restoration_indices : Variable(torch.LongTensor)
        Indices into the sorted_tensor such that
        ``sorted_tensor.index_select(0, restoration_indices) == original_tensor``
    """

    if not isinstance(tensor, Variable) or not isinstance(sequence_lengths, Variable):
        raise ValueError("Both the tensor and sequence lengths must be torch.autograd.Variables.")

    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)
    # This is ugly, but required - we are creating a new variable at runtime, so we
    # must ensure it has the correct CUDA vs non-CUDA type. We do this by cloning and
    # refilling one of the inputs to the function.
    index_range = sequence_lengths.data.clone().copy_(torch.arange(0, len(sequence_lengths)))
    # This is the equivalent of zipping with index, sorting by the original
    # sequence lengths and returning the now sorted indices.
    index_range = Variable(index_range.long())
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_sequence_lengths, restoration_indices


class SelfAttentiveSum(nn.Module):
    """
    Attention mechanism to get a weighted sum of RNN output sequence to a single RNN output dimension.
    """
    def __init__(self, embed_dim, hidden_dim):
        """
        :param embed_dim: in forward(input_embed), the size will be batch x seq_len x emb_dim
        :param hidden_dim:
        """
        super(SelfAttentiveSum, self).__init__()
        self.key_maker = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.key_rel = nn.ReLU()
        self.hidden_dim = hidden_dim
        self.key_output = nn.Linear(hidden_dim, 1, bias=False)
        self.key_softmax = nn.Softmax(dim=1)

    def forward(self, input_embed):     # batch x seq_len x emb_dim
        input_embed_squeezed = input_embed.view(-1, input_embed.size()[2])  # batch * seq_len x emb_dim
        k_d = self.key_maker(input_embed_squeezed)      # batch * seq_len x hidden_dim
        k_d = self.key_rel(k_d)
        if self.hidden_dim == 1:
            k = k_d.view(input_embed.size()[0], -1)     # batch x seq_len
        else:
            k = self.key_output(k_d).view(input_embed.size()[0], -1)  # (batch_size, seq_length)
        weighted_keys = self.key_softmax(k).view(input_embed.size()[0], -1, 1)  # batch x seq_len x 1
        weighted_values = torch.sum(weighted_keys * input_embed, 1)  # batch_size, embed_dim
        return weighted_values, weighted_keys


class CharEncoder(nn.Module):
    def __init__(self, char_vocab, args):
        super(CharEncoder, self).__init__()
        conv_dim_input = 100
        filters = 5
        self.char_W = nn.Embedding(char_vocab.size(), conv_dim_input, padding_idx=0)
        self.conv1d = nn.Conv1d(conv_dim_input, args.char_emb_size, filters)  # input, output, filter_number

    def forward(self, span_chars):
        char_embed = self.char_W(span_chars).transpose(1, 2)  # [batch_size, char_embedding, max_char_seq]
        conv_output = [self.conv1d(char_embed)]  # list of [batch_size, filter_dim, max_char_seq, filter_number]
        conv_output = [F.relu(c) for c in conv_output]  # batch_size, filter_dim, max_char_seq, filter_num
        cnn_rep = [F.max_pool1d(i, i.size(2)) for i in conv_output]  # batch_size, filter_dim, 1, filter_num
        cnn_output = torch.squeeze(torch.cat(cnn_rep, 1), 2)  # batch_size, filter_num * filter_dim, 1
        return cnn_output
