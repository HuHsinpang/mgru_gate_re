"""Define the neural network, loss function"""
import numpy as np
import torch
import math
import copy
import torch.nn as nn
import threading
import torch.nn.functional as F
from fastNLP import seq_len_to_mask
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
# from .Focal_Loss import focal_loss


class CNN(nn.Module):
    def __init__(self, data_loader, params):
        super(CNN, self).__init__()
        # word and position embedding layer
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=data_loader.embedding_vectors, freeze=False)
        self.pos1_embedding = nn.Embedding(
            params.pos_dis_limit * 2 + 3, params.pos_emb_dim)
        self.pos2_embedding = nn.Embedding(
            params.pos_dis_limit * 2 + 3, params.pos_emb_dim)

        self.max_len = params.max_len
        # dropout layer
        self.dropout = nn.Dropout(params.dropout_ratio)

        feature_dim = params.word_emb_dim + params.pos_emb_dim * 2
        # encode sentence level features via cnn
        self.covns = nn.ModuleList([nn.Sequential(
            nn.Conv1d(in_channels=feature_dim,
                      out_channels=params.filter_num, kernel_size=k),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=self.max_len-k+1)) for k in params.filters])

        filter_dim = params.filter_num * len(params.filters)
        labels_num = len(data_loader.label2idx)
        # output layer
        self.linear = nn.Linear(filter_dim, labels_num)
        self.loss = nn.CrossEntropyLoss()

        # 使用focal loss作为损失函数，但是实际应用中效果并不好
        # labels_ratio = F.softmax(torch.tensor([label_num / 5000 for label_num in data_loader.label_nums]),
        # 						 dim=-1).numpy().tolist()
        # self.loss = focal_loss(alpha=labels_ratio, gamma=2, num_classes=labels_num)

    def forward(self, x):
        batch_sents = x['sents']
        batch_pos1s = x['pos1s']
        batch_pos2s = x['pos2s']
        word_embs = self.word_embedding(batch_sents)
        pos1_embs = self.pos1_embedding(batch_pos1s)
        pos2_embs = self.pos2_embedding(batch_pos2s)

        # batch_size x seq_len x feature_dim
        input_feature = torch.cat([word_embs, pos1_embs, pos2_embs], dim=2)
        input_feature = input_feature.permute(
            0, 2, 1) 	# (batch_size,feature_dim,seq_len)
        input_feature = self.dropout(input_feature)

        out = [conv(input_feature)
               for conv in self.covns] 	# (batch_size,filter_num,1)
        out = torch.cat(out, dim=1)
        out = self.dropout(out)
        # (batch_size, (filter_num*window_num))
        out = out.view(-1, out.size(1))

        out = self.linear(self.dropout(out))
        return out


class BiLSTM_Att(nn.Module):
    def __init__(self, data_loader, params):
        super(BiLSTM_Att, self).__init__()
        self.word_embedding = nn.Embedding.from_pretrained(
            data_loader.embedding_vectors, freeze=False)

        self.out_size = len(data_loader.label2idx)
        self.hidden_dim = params.hidden_dim
        self.batch_size = params.batch_size
        # self.feature_dim = params.word_emb_dim + params.pos_emb_dim * 2
        self.feature_dim = params.word_emb_dim

        self.lstm = nn.LSTM(
            self.feature_dim, self.hidden_dim, bidirectional=True)
        # self.att_weight = nn.Parameter(torch.Tensor(self.batch_size, 1, self.hidden_dim))
        self.att_weight = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        self.dropout_emb = nn.Dropout(p=0.5)
        self.dropout_lstm = nn.Dropout(p=0.5)
        self.dropout_att = nn.Dropout(p=0.5)

        self.dense = nn.Linear(self.hidden_dim, self.out_size)
        self.loss = nn.CrossEntropyLoss()

    def attention(self, H, batch_mask):
        batch_mask = torch.unsqueeze(batch_mask, 1)
        M = torch.tanh(H)
        a = torch.bmm(self.att_weight.repeat(self.batch_size, 1, 1), M)
        a = a.masked_fill(batch_mask.eq(0), float(
            '-inf'))  # (batch_size,1,seq_len)
        att_score = F.softmax(a, dim=2).transpose(
            1, 2)  # (batch_size,seq_len,1)
        return torch.bmm(H, att_score)  # (batch_size,hidden_dim,1)

    def forward(self, X):
        batch_sents = X['sents']
        # batch_pos1s = X['pos1s']
        # batch_pos2s = X['pos2s']
        batch_mask = X['mask']
        batch_lens = X['lens']

        word_embs = self.word_embedding(batch_sents)
        # pos1_embs = self.pos1_embedding(batch_pos1s)
        # pos2_embs = self.pos2_embedding(batch_pos2s)

        # input_feature = torch.cat([word_embs, pos1_embs, pos2_embs], dim=2).transpose(0, 1)
        input_feature = word_embs.transpose(0, 1)
        sents_represent = pack_padded_sequence(
            input=self.dropout_emb(input_feature), lengths=np.array(batch_lens))

        # (seq_len,batch_size,vector_size)
        # lstm_out : (seq_len,batch_size,hidden_dim*2)
        lstm_out, state = self.lstm(sents_represent)
        # batch_size, seq_len, hidden_dim
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = lstm_out.transpose(1, 2).view(
            self.batch_size, self.hidden_dim, 2, -1)		# (batch_size,hidden_dim,seq_len)
        lstm_out = self.dropout_lstm(torch.sum(lstm_out, dim=2))
        # (batch_size,hidden_dim,1)
        att_out = torch.tanh(self.attention(lstm_out, batch_mask))
        att_out = self.dropout_att(att_out)
        out = self.dense(torch.squeeze(att_out, 2))  # 经过一个全连接矩阵 W*h + b
        return out


class BiLSTM_MaxPooling(nn.Module):
    def __init__(self, data_loader, params):
        super(BiLSTM_MaxPooling, self).__init__()

        self.out_size = len(data_loader.label2idx)
        self.hidden_dim = params.hidden_dim
        self.batch_size = params.batch_size
        self.feature_dim = params.word_emb_dim + params.pos_emb_dim * 2
        self.lstm = nn.LSTM(
            self.feature_dim, self.hidden_dim//2, bidirectional=True)

        self.word_embedding = nn.Embedding.from_pretrained(
            data_loader.embedding_vectors, freeze=False)
        self.pos1_embedding = nn.Embedding(
            params.pos_dis_limit * 2 + 3, params.pos_emb_dim)
        self.pos2_embedding = nn.Embedding(
            params.pos_dis_limit * 2 + 3, params.pos_emb_dim)

        self.att_weight = nn.Parameter(torch.randn(
            (self.batch_size, 1, self.hidden_dim)))

        self.dense = nn.Linear(self.hidden_dim, self.out_size)
        self.device = None
        self.loss = nn.CrossEntropyLoss()
        if params.gpu >= 0:
            self.device = self.cuda(device=params.gpu)

    def begin_state(self):
        state = (
            torch.zeros(2, self.batch_size, self.hidden_dim // 2),
            torch.zeros(2, self.batch_size, self.hidden_dim // 2))
        if self.device:
            return state.to(self.device)
        else:
            return state

    def forward(self, X):
        batch_sents = X['sents']
        batch_pos1s = X['pos1s']
        batch_pos2s = X['pos2s']

        word_embs = self.word_embedding(batch_sents)
        pos1_embs = self.pos1_embedding(batch_pos1s)
        pos2_embs = self.pos2_embedding(batch_pos2s)

        input_feature = torch.cat(
            [word_embs, pos1_embs, pos2_embs], dim=2).transpose(0, 1)
        # list_out : (seq_len,batch_size,hidden_dim)
        lstm_out, state = self.lstm(input_feature, self.begin_state())
        out, _ = torch.max(lstm_out, dim=0)  # (1,batch_size,hidden_dim)
        out = self.dense(out.squeeze(0))  # 经过一个全连接矩阵 W*h + b
        return out


# ************************************************ Transformer *******************************************************
def get_embedding(max_seq_len, embedding_dim, padding_idx=None, rel_pos_init=0):
    """Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    rel pos init:
    如果是0，那么从-max_len到max_len的相对位置编码矩阵就按0-2*max_len来初始化，
    如果是1，那么就按-max_len,max_len来初始化
    """
    num_embeddings = 2*max_seq_len+1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    if rel_pos_init == 0:
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1) * emb.unsqueeze(0)
    else:
        emb = torch.arange(-max_seq_len, max_seq_len+1,
                           dtype=torch.float).unsqueeze(1)*emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)],
                    dim=1).view(num_embeddings, -1)
    if embedding_dim % 2 == 1:
        # zero pad
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0
    return emb


class Absolute_Position_Embedding(nn.Module):
    def __init__(self, fusion_func, hidden_size, learnable, pos_norm=False, max_len=51):
        '''

        :param hidden_size:
        :param max_len:
        :param learnable:
        :param debug:
        :param fusion_func:暂时只有add和concat(直接拼接然后接线性变换)，后续得考虑直接拼接再接非线性变换
        '''
        super().__init__()
        self.fusion_func = fusion_func
        self.mode = 1
        assert ('add' in self.fusion_func) != ('concat' in self.fusion_func)
        if 'add' in self.fusion_func:
            self.fusion_func = 'add'
        else:
            self.fusion_func = 'concat'
        # 备注，在SE绝对位置里，会需要nonlinear操作来融合两种位置pos，但普通的不需要，所以只根据字符串里有哪个关键字来判断
        self.pos_norm = pos_norm
        self.hidden_size = hidden_size
        pe = get_embedding(max_len, hidden_size)

        pe_sum = pe.sum(dim=-1, keepdim=True)
        if self.pos_norm:
            with torch.no_grad():
                pe = pe / pe_sum
        pe = pe.unsqueeze(0)
        self.pe = nn.Parameter(pe, requires_grad=learnable)

        if self.fusion_func == 'concat':
            self.proj = nn.Linear(self.hidden_size * 2, self.hidden_size)

        if self.mode:
            print('position embedding:')
            print(self.pe[:100])
            print('pe size:{}'.format(self.pe.size()))
            print('pe avg:{}'.format(torch.sum(self.pe) /
                                          (self.pe.size(2)*self.pe.size(1))))

    def forward(self, inp):
        batch = inp.size(0)
        if self.fusion_func == 'add':
            output = inp + self.pe[:, :inp.size(1)]
        elif self.fusion_func == 'concat':
            inp = torch.cat([inp, self.pe[:, :inp.size(1)].repeat(
                [batch]+[1]*(inp.dim()-1))], dim=-1)
            output = self.proj(inp)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, pe, scaled=True, drop=None, dvc=None,
                 k_proj=False, q_proj=False, v_proj=False, r_proj=False):
        '''

        :param hidden_size:
        :param num_heads:
        :param scaled:
        :param debug:
        :param device:
        '''
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.per_head_size = self.hidden_size // self.num_heads
        self.scaled = scaled
        if dvc is None:
            dvc = torch.device('cpu')
        self.dvc = dvc
        self.pe = pe

        self.k_proj = k_proj
        self.q_proj = q_proj
        self.v_proj = v_proj
        self.r_proj = r_proj

        self.w_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.w_r = nn.Linear(self.hidden_size, self.hidden_size)
        self.pos_drop = nn.Dropout(drop["posi"])
        self.u = nn.Parameter(torch.Tensor(self.num_heads, self.per_head_size))
        self.v = nn.Parameter(torch.Tensor(self.num_heads, self.per_head_size))

        self.att_drop = nn.Dropout(drop["attn"])

        self.activate = nn.GELU()

        self.res_drop = nn.Dropout(drop["ff"])

        self.ff_final = nn.Linear(self.hidden_size, self.hidden_size)
        self.ff_drop = nn.Dropout(drop["ff"])
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, src, seq_len):
        # B prepare relative position encoding
        max_len = max(seq_len)
        rel_distance = self.seq_len_to_rel_distance(max_len)
        rel_distance_flat = rel_distance.view(-1)
        rel_pos_embedding_flat = self.pe[rel_distance_flat+max_len]
        # rel_pos_embedding_flat = self.pe[rel_distance_flat + self.max_seq_len]
        rel_pos_embedding = rel_pos_embedding_flat.view(
            size=[max_len, max_len, self.hidden_size])
        # E prepare relative position encoding

        if self.k_proj:
            key = self.w_k(src)
        else:
            key = src
        if self.q_proj:
            query = self.w_q(src)
        else:
            query = src
        if self.v_proj:
            value = self.w_v(src)
        else:
            value = src
        if self.r_proj:
            rel_pos_embedding = self.w_r(self.pos_drop(rel_pos_embedding))

        batch = key.size(0)
        max_len = key.size(1)

        # batch * seq_len * n_head * d_head
        key = torch.reshape(
            key, [batch, max_len, self.num_heads, self.per_head_size])
        query = torch.reshape(
            query, [batch, max_len, self.num_heads, self.per_head_size])
        value = torch.reshape(
            value, [batch, max_len, self.num_heads, self.per_head_size])
        rel_pos_embedding = torch.reshape(rel_pos_embedding,
                                          [max_len, max_len, self.num_heads, self.per_head_size])

        # batch * n_head * seq_len * d_head
        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)

        # batch * n_head * d_head * key_len
        key = key.transpose(-1, -2)

        # A
        A_ = torch.matmul(query, key)

        # B
        rel_pos_embedding_for_b = rel_pos_embedding.unsqueeze(
            0).permute(0, 3, 1, 4, 2)
        # after above, rel_pos_embedding: batch * num_head * query_len * per_head_size * key_len
        query_for_b = query.view(
            [batch, self.num_heads, max_len, 1, self.per_head_size])
        # print('query for b:{}'.format(query_for_b.size()))
        # print('rel_pos_embedding_for_b{}'.format(rel_pos_embedding_for_b.size()))
        B_ = torch.matmul(query_for_b, rel_pos_embedding_for_b).squeeze(-2)

        # D
        rel_pos_embedding_for_d = rel_pos_embedding.unsqueeze(-2)
        # after above, rel_pos_embedding: query_seq_len * key_seq_len * num_heads * 1 *per_head_size
        v_for_d = self.v.unsqueeze(-1)
        # v_for_d: num_heads * per_head_size * 1
        D_ = torch.matmul(rel_pos_embedding_for_d,
                          v_for_d).squeeze(-1).squeeze(-1).permute(2, 0, 1).unsqueeze(0)

        # C
        # key: batch * n_head * d_head * key_len
        u_for_c = self.u.unsqueeze(0).unsqueeze(-2)
        C_ = torch.matmul(u_for_c, key)

        attn_score_raw = A_ + B_ + C_ + D_

        if self.scaled:
            attn_score_raw = attn_score_raw / math.sqrt(self.per_head_size)

        mask = seq_len_to_mask(torch.Tensor(seq_len)).bool().unsqueeze(1).unsqueeze(1).to(attn_score_raw.device)
        attn_score_raw_masked = attn_score_raw.masked_fill(~mask, -1e15)

        attn_score = F.softmax(attn_score_raw_masked, dim=-1)
        attn_score = self.att_drop(attn_score)

        value_weighted_sum = torch.matmul(attn_score, value)

        res_x = value_weighted_sum.transpose(1, 2).contiguous(). \
            reshape(batch, max_len, self.hidden_size)

        result = src + self.res_drop(res_x)
        return self.activate(self.layer_norm(result))
        # result = self.ff_final(self.ff_drop(self.activate(self.layer_norm(result))))
        # return self.layer_norm(result + self.res_drop(result))
        # return self.layer_norm(res_x + self.activate(self.res_drop(res_x)))

    def seq_len_to_rel_distance(self, max_seq_len):
        '''

        :param seq_len: seq_len batch
        :return: L*L rel_distance
        '''
        index = torch.arange(0, max_seq_len)
        assert index.size(0) == max_seq_len
        assert index.dim() == 1
        index = index.repeat(max_seq_len, 1)
        offset = torch.arange(0, max_seq_len).unsqueeze(1)
        offset = offset.repeat(1, max_seq_len)
        index = index - offset
        index = index.to(self.dvc)
        return index


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Transformer_Encoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, emb_dim, hidden_dim, encoder_layer, num_layers, norm=None):
        super(Transformer_Encoder, self).__init__()
        self.map = nn.Linear(emb_dim, hidden_dim)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, lens=None, src_key_padding_mask=None):
        output = self.map(src)

        for mod in self.layers:
            output = mod(output, seq_len=lens)

        if self.norm is not None:
            output = self.norm(output)

        return output


class Transformer_GateAttention(nn.Module):
    def __init__(self, data_loader, params):
        super(Transformer_GateAttention, self).__init__()
        emb_dim = params.word_emb_dim
        hidden_dim = params.hidden_dim
        max_len = params.max_len
        head = params.head
        drop_dict = {"embed":0.4, "output":0.3, "posi":0.2, "post":0.3, "ff":0.15, "ff_2":0, "attn":0.15}
        self.batch_size = params.batch_size

        # 词向量与相对位置向量
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=data_loader.embedding_vectors, freeze=False)
        self.pos1_embedding = nn.Embedding(
            params.pos_dis_limit * 2 + 3, params.pos_emb_dim)
        self.pos2_embedding = nn.Embedding(
            params.pos_dis_limit * 2 + 3, params.pos_emb_dim)
        self.dropout_emb = nn.Dropout(p=drop_dict["embed"])

        self.use_abs_pos = params.use_abs_pos
        if self.use_abs_pos:
            self.pos_encode = Absolute_Position_Embedding('nonlinear_add', emb_dim, learnable=False, pos_norm=False)
        pe = get_embedding(max_len, hidden_dim, rel_pos_init=1)
        pe_sum = pe.sum(dim=-1, keepdim=True)
        if params.pos_norm:
            with torch.no_grad():
                pe = pe/pe_sum
        self.pe = nn.Parameter(pe, requires_grad=False)

        encoder_layer = EncoderLayer(hidden_dim, head, pe=self.pe,
                                     scaled=True, drop=drop_dict, dvc=data_loader.compute_device)
        self.encoder = Transformer_Encoder(emb_dim, hidden_dim, encoder_layer, num_layers=1)
        self.encoder_drop = nn.Dropout(drop_dict["output"])

        self.att_weight = nn.Parameter(torch.randn(1, hidden_dim, 1))
        self.dropout_att = nn.Dropout(p=0.4)

        self.dense = nn.Linear(hidden_dim, len(data_loader.label2idx))
        self.loss = nn.CrossEntropyLoss()

    def attention(self, H, batch_mask):
        batch_mask = torch.unsqueeze(
            batch_mask, 1)     # batch_size, 1, seq_len
        M = torch.tanh(H)
        # (batch_size, seq_len, hidden) * (batch_size, hidden, 1)
        a = torch.bmm(M, self.att_weight.repeat(
            self.batch_size, 1, 1)).transpose(1, 2)
        a = a.masked_fill(batch_mask.eq(0), float(
            '-inf'))  # batch_size, 1, seq_len
        att_score = F.softmax(a, dim=-1)
        # (batch_size, hidden_dim, seq_len) X (batch_size, seq_len, 1)
        return torch.bmm(att_score, H)  # (batch_size,1,hidden_dim)

    def forward(self, X):
        torch.set_printoptions(profile="full")
        batch_sents = X['sents']
        batch_pos1s = X['pos1s']
        batch_pos2s = X['pos2s']
        batch_mask = X['mask']      # batch_size, max_seq_len
        batch_lens = X['lens']

        # max_seq_len = batch_sents.size(1)

        # 词向量
        word_embs = self.word_embedding(batch_sents)
        pos1_embs = self.pos1_embedding(batch_pos1s)
        pos2_embs = self.pos2_embedding(batch_pos2s)
        word_embedding = torch.cat([word_embs, pos1_embs, pos2_embs], dim=2)
        if self.use_abs_pos:
            word_embedding = self.pos_encode(word_embedding)
        word_embedding = self.dropout_emb(word_embedding)

        encoded_sents = self.encoder(word_embedding, batch_lens)
        encoded_sents = self.encoder_drop(encoded_sents)

        att_out = torch.tanh(self.attention(encoded_sents, batch_mask))
        att_out = self.dropout_att(att_out)
        out = self.dense(torch.squeeze(att_out, 1))  # 经过一个全连接矩阵 W*h + b
        return out


# *************************************************** mogLSTM *******************************************************
class MogLSTM(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 bias: bool = True,
                 mog_iterations: int = 4,
                 batch_first: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.mog_iterations = mog_iterations
        self.batch_first = batch_first

        self.Q = nn.Parameter(torch.Tensor(2, hidden_size, input_size))
        self.R = nn.Parameter(torch.Tensor(2, input_size, hidden_size))

        self.fw_lstmcell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=bias).cuda()
        self.bw_lstmcell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size, bias=bias).cuda()

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def mogrify(self, xt, ht, Q, R):
        xt = (2 * torch.sigmoid(ht @ Q)) * xt  # i=1
        ht = (2 * torch.sigmoid(xt @ R)) * ht  # i=2
        xt = (2 * torch.sigmoid(ht @ Q)) * xt  # i=3
        ht = (2 * torch.sigmoid(xt @ R)) * ht  # i=4
        return xt, ht

    def moglstmCell(self, batch_sizes, input_x_tensor, ht_tensor, direction: int = 0):
        if not direction:  # ********************** forword *************************
            max_batch_size = int(batch_sizes[0])
            fw_minibatch_start = 0

            Ct_fw = torch.zeros((max_batch_size, self.hidden_size)).cuda()
            ht_pre_temp_fw = torch.zeros((max_batch_size, self.hidden_size)).cuda()

            for t, mini_batch_fw in enumerate(batch_sizes):  # seq_len
                fw_minibatch_end = fw_minibatch_start + mini_batch_fw
                xt = input_x_tensor[fw_minibatch_start:fw_minibatch_end]  # forward_xt  (fw_mini_batch, input_sz)

                if t == 0:
                    ht_pre_temp_fw[:mini_batch_fw], Ct_fw[:mini_batch_fw] = self.fw_lstmcell(xt)
                else:
                    ht_pre = ht_pre_temp_fw[:mini_batch_fw].clone()  # forward_ht  (fw_mini_batch, hidden_sz)
                    Ct_pre = Ct_fw[:mini_batch_fw].clone()  # forward_Ct  (fw_mini_batch, hidden_sz)
                    xt, ht_pre = self.mogrify(xt, ht_pre, self.Q[0], self.R[0])  # out:(mini_batch, input_sz), (mini_batch, hidden_sz)

                    ht_pre_temp_fw[:mini_batch_fw], Ct_fw[:mini_batch_fw] = self.fw_lstmcell(xt, (ht_pre, Ct_pre))
                ht_tensor[fw_minibatch_start:fw_minibatch_end] = ht_pre_temp_fw[:mini_batch_fw]
                fw_minibatch_start = fw_minibatch_end
        else:  # ********************** backword ************************
            max_batch_size = int(batch_sizes[-1])
            bw_minibatch_end = input_x_tensor.size(0)

            Ct_bw = torch.zeros((max_batch_size, self.hidden_size)).cuda()
            ht_pre_temp_bw = torch.zeros((max_batch_size, self.hidden_size)).cuda()

            for t, mini_batch_bw in enumerate(batch_sizes):  # seq_len
                bw_minibatch_start = bw_minibatch_end - mini_batch_bw
                xt = input_x_tensor[bw_minibatch_start:bw_minibatch_end]  # backward_xt (bw_mini_batch, input_sz)

                if t == 0:
                    ht_pre_temp_bw[:mini_batch_bw], Ct_bw[:mini_batch_bw] = self.bw_lstmcell(xt)
                else:
                    ht_pre = ht_pre_temp_bw[:mini_batch_bw].clone()  # backward_ht (bw_mini_batch, hidden_sz)
                    Ct_pre = Ct_bw[:mini_batch_bw].clone()  # backward_Ct (bw_mini_batch, hidden_sz)
                    xt, ht_pre = self.mogrify(xt, ht_pre, self.Q[1], self.R[1])  # out:(mini_batch, input_sz), (mini_batch, hidden_sz)

                    ht_pre_temp_bw[:mini_batch_bw], Ct_bw[:mini_batch_bw] = self.bw_lstmcell(xt, (ht_pre, Ct_pre))
                ht_tensor[bw_minibatch_start:bw_minibatch_end] = ht_pre_temp_bw[:mini_batch_bw]
                bw_minibatch_end = bw_minibatch_start

    def forward(self, x: torch.Tensor, init_states=None):
        x, batch_sizes, sorted_indices, unsorted_indices = x

        # ht and Ct: Ct和ht存储t这一时刻状态，要及时更新
        assert init_states is None
        ht_fw = torch.zeros((x.shape[0], self.hidden_size)).cuda()
        ht_bw = torch.zeros((x.shape[0], self.hidden_size)).cuda()

        fw_moglstm_cell = threading.Thread(target=self.moglstmCell, args=(batch_sizes, x, ht_fw, 0))
        bw_moglstm_cell = threading.Thread(target=self.moglstmCell, args=(torch.flip(batch_sizes, [0]), x, ht_bw, 1))
        fw_moglstm_cell.start()
        bw_moglstm_cell.start()
        fw_moglstm_cell.join()
        bw_moglstm_cell.join()

        hidden_seq = PackedSequence(
            torch.add(ht_fw, ht_bw).contiguous(),
            batch_sizes, sorted_indices,
            unsorted_indices)  # pack_seq_len, 2*hidden_sz

        return hidden_seq, (None, None)


class MogGRU(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 bias: bool = True,
                 mog_iterations: int = 4,
                 batch_first: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.mog_iterations = mog_iterations
        self.batch_first = batch_first

        self.Q = nn.Parameter(torch.Tensor(2, hidden_size, input_size))
        self.R = nn.Parameter(torch.Tensor(2, input_size, hidden_size))

        self.fw_grucell = nn.GRUCell(input_size=input_size, hidden_size=hidden_size, bias=bias).cuda()
        self.bw_grucell = nn.GRUCell(input_size=input_size, hidden_size=hidden_size, bias=bias).cuda()

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def mogrify(self, xt, ht, Q, R):
        xt = (2 * torch.sigmoid(ht @ Q)) * xt  # i=1
        ht = (2 * torch.sigmoid(xt @ R)) * ht  # i=2
        xt = (2 * torch.sigmoid(ht @ Q)) * xt  # i=3
        ht = (2 * torch.sigmoid(xt @ R)) * ht  # i=4
        return xt, ht

    def mogGRUCell(self, batch_sizes, input_x_tensor, ht_tensor, direction: int = 0):
        if not direction:  # ********************** forword *************************
            max_batch_size = int(batch_sizes[0])
            fw_minibatch_start = 0

            ht_pre_temp_fw = torch.zeros((max_batch_size, self.hidden_size)).cuda()

            for t, mini_batch_fw in enumerate(batch_sizes):  # seq_len
                fw_minibatch_end = fw_minibatch_start + mini_batch_fw
                xt = input_x_tensor[fw_minibatch_start:fw_minibatch_end]  # forward_xt  (fw_mini_batch, input_sz)

                if t == 0:
                    ht_pre_temp_fw[:mini_batch_fw] = self.fw_grucell(xt)
                else:
                    ht_pre = ht_pre_temp_fw[:mini_batch_fw].clone()  # forward_ht  (fw_mini_batch, hidden_sz)
                    xt, ht_pre = self.mogrify(xt, ht_pre, self.Q[0], self.R[0])  # out:(mini_batch, input_sz), (mini_batch, hidden_sz)

                    ht_pre_temp_fw[:mini_batch_fw] = self.fw_grucell(xt, ht_pre)
                ht_tensor[fw_minibatch_start:fw_minibatch_end] = ht_pre_temp_fw[:mini_batch_fw]
                fw_minibatch_start = fw_minibatch_end
        else:  # ********************** backword ************************
            max_batch_size = int(batch_sizes[-1])
            bw_minibatch_end = input_x_tensor.size(0)

            ht_pre_temp_bw = torch.zeros((max_batch_size, self.hidden_size)).cuda()

            for t, mini_batch_bw in enumerate(batch_sizes):  # seq_len
                bw_minibatch_start = bw_minibatch_end - mini_batch_bw
                xt = input_x_tensor[bw_minibatch_start:bw_minibatch_end]  # backward_xt (bw_mini_batch, input_sz)

                if t == 0:
                    ht_pre_temp_bw[:mini_batch_bw] = self.bw_grucell(xt)
                else:
                    ht_pre = ht_pre_temp_bw[:mini_batch_bw].clone()  # backward_ht (bw_mini_batch, hidden_sz)
                    xt, ht_pre = self.mogrify(xt, ht_pre, self.Q[1], self.R[1])  # out:(mini_batch, input_sz), (mini_batch, hidden_sz)

                    ht_pre_temp_bw[:mini_batch_bw] = self.bw_grucell(xt, ht_pre)
                ht_tensor[bw_minibatch_start:bw_minibatch_end] = ht_pre_temp_bw[:mini_batch_bw]
                bw_minibatch_end = bw_minibatch_start

    def forward(self, x: torch.Tensor, init_states=None):
        x, batch_sizes, sorted_indices, unsorted_indices = x

        # ht and Ct: Ct和ht存储t这一时刻状态，要及时更新
        assert init_states is None
        ht_fw = torch.zeros((x.shape[0], self.hidden_size)).cuda()
        ht_bw = torch.zeros((x.shape[0], self.hidden_size)).cuda()

        fw_moglstm_cell = threading.Thread(target=self.mogGRUCell, args=(batch_sizes, x, ht_fw, 0))
        bw_moglstm_cell = threading.Thread(target=self.mogGRUCell, args=(torch.flip(batch_sizes, [0]), x, ht_bw, 1))
        fw_moglstm_cell.start()
        bw_moglstm_cell.start()
        fw_moglstm_cell.join()
        bw_moglstm_cell.join()

        hidden_seq = PackedSequence(
            torch.add(ht_fw, ht_bw).contiguous(),
            batch_sizes, sorted_indices,
            unsorted_indices)  # pack_seq_len, 2*hidden_sz

        return hidden_seq, (None, None)


class Mog(nn.Module):
    def __init__(self, data_loader, params):
        super(Mog, self).__init__()
        emb_dim = params.word_emb_dim
        self.hidden_dim = params.hidden_dim
        self.batch_size = params.batch_size
        self.beta = params.beta

        drop_dict = {"embed":0.55, "output":0.55, "attn":0.5}

        # 词向量与相对位置向量
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=data_loader.embedding_vectors, freeze=False)
        self.pos1_embedding = nn.Embedding(
            params.pos_dis_limit * 2 + 3, params.pos_emb_dim)
        self.pos2_embedding = nn.Embedding(
            params.pos_dis_limit * 2 + 3, params.pos_emb_dim)
        self.dropout_emb = nn.Dropout(p=drop_dict["embed"])

        # self.encoder = MogLSTM(emb_dim, self.hidden_dim, batch_first=True)
        self.encoder = MogGRU(emb_dim, self.hidden_dim, batch_first=True)
        self.encoder_drop = nn.Dropout(drop_dict["output"])

        self.entity_w = nn.Linear(self.hidden_dim, 1)

        self.att_weight = nn.Parameter(torch.randn(1, self.hidden_dim, 1))
        self.dropout_att = nn.Dropout(drop_dict["attn"])

        self.dense = nn.Linear(self.hidden_dim, len(data_loader.label2idx))
        self.loss = nn.CrossEntropyLoss()

    def attention(self, H):
        M = torch.tanh(H)
        # (batch_size, seq_len, hidden) * (batch_size, hidden, 1)
        a = torch.bmm(M, self.att_weight.repeat(self.batch_size, 1, 1)).squeeze()
        a = a.masked_fill(a.bool().eq(0), float('-inf'))
        att_score = F.softmax(a, dim=-1)
        return torch.bmm(att_score.unsqueeze(1), H)     # (batch_size, 1, seq_len) (batch_size, seq_len, hidden_dim)

    # 阈值策略.归一化，留下超过多大关系的值
    def gate_attention(self, encoded_sents, entities, slice_masks, entity_masks, slice_lens, beta=0.1):
        entity_weight = self.entity_w(entities).squeeze().\
            masked_fill(entity_masks.eq(0), float('-inf'))   # out:(b,e_len)
        # entity_weight = entities.sum(dim=-1).squeeze().masked_fill(entity_masks.eq(0), float('-inf'))
        entity_weight = F.softmax(entity_weight, dim=-1)
        entity = torch.bmm(entity_weight.unsqueeze(1), entities)     # (b,1,e_len)@(b,e_len,hidden)->(b,1,hidden)
        p = torch.tanh(entity).transpose(1, 2)

        words = torch.bmm(encoded_sents, p).squeeze().masked_fill(slice_masks.eq(0), float('-inf'))     # (batch,len,hidden)@(batch,hidden,1)
        word_weight = F.softmax(words, dim=-1)

        word_weight = torch.mul(slice_lens.view(-1, 1), word_weight)     # b1*bl->bl
        rela = torch.div(F.threshold(word_weight, beta, 0.), slice_lens.view(-1, 1))
        norm_ratio = torch.max(rela, 1)[0]
        rela = torch.div(rela, norm_ratio.view(-1, 1))
        return torch.cat([entities, torch.mul(rela.unsqueeze(2), encoded_sents)], dim=1)      # bl1*blh->blh

    # 截断策略.设定保留个数, 0=beta<=6 version 2
    # def gate_attention(self, encoded_sents, entities, slice_masks, entity_masks, slice_lens, beta=0.1):
    #     beta = int(beta)
    #     entity_weight = self.entity_w(entities).squeeze(). \
    #         masked_fill(entity_masks.eq(0), float('-inf'))  # out:(b,e_len)
    #     entity_weight = F.softmax(entity_weight, dim=-1)
    #     entity = torch.bmm(entity_weight.unsqueeze(1), entities)  # (b,1,e_len)@(b,e_len,hidden)->(b,1,hidden)
    #     p = torch.tanh(entity).transpose(1, 2)
    
    #     words = torch.bmm(encoded_sents, p).squeeze().masked_fill(slice_masks.eq(0),
    #                                                               float('-inf'))  # (batch,len,hidden)@(batch,hidden,1)
    #     sorted_word_weight, indices = F.softmax(words, dim=-1).sort(dim=-1, descending=True)    # bl
    #     sorted_word_weight = torch.mul(slice_lens.view(-1, 1), sorted_word_weight)
    
    #     sorted_word_weight[:, beta:] = 0
    #     rela = torch.div(sorted_word_weight, sorted_word_weight.max(dim=1, keepdim=True)[0])
    #     indices = indices.unsqueeze(2).repeat(1, 1, self.hidden_dim)
    #     encoded_sents = torch.gather(encoded_sents, 1, indices)
    
    #     return torch.cat([entities, torch.mul(rela.unsqueeze(2), encoded_sents)], dim=1)  # bl1*blh->blh

    def forward(self, X):
        torch.set_printoptions(profile="full")
        batch_sents = X['sents']
        batch_entities = X['entities']
        batch_slices = X['slices']
        batch_slice_lens = X['slice_lens']
        batch_entity_masks = X['entity_masks']
        batch_slice_masks = X['slice_masks']
        # batch_mask = X['mask']      # batch_size, max_seq_len
        batch_lens = X['lens']

        # 词向量
        word_embs = self.word_embedding(batch_sents)
        word_represent = pack_padded_sequence(input=self.dropout_emb(word_embs),
                                              lengths=np.array(batch_lens),
                                              batch_first=True)

        packed_sents, _ = self.encoder(word_represent)
        encoded_sents, _ = pad_packed_sequence(packed_sents, batch_first=True)
        encoded_sents = self.encoder_drop(encoded_sents)

        batch_entities = batch_entities.view(self.batch_size, -1, 1).repeat(1, 1, self.hidden_dim)
        batch_slices = batch_slices.view(self.batch_size, -1, 1).repeat(1, 1, self.hidden_dim)
        encoded_entities = torch.gather(encoded_sents, 1, batch_entities)
        encoded_slices = torch.gather(encoded_sents, 1, batch_slices)
        encoded_sents = self.gate_attention(
            encoded_slices, encoded_entities, batch_slice_masks, batch_entity_masks, batch_slice_lens, self.beta)

        att_out = torch.tanh(self.attention(encoded_sents)).squeeze()
        out = self.dense(self.dropout_att(att_out))  # 经过一个全连接矩阵 W*h + b
        return out

# 阈值策略beta:
# 1.13, precison: 82.34; recall: 85.68; f1: 83.98
# 1.12, precison: 83.50; recall: 84.65; f1: 84.07   precison: 82.73; recall: 86.08; f1: 84.37
# 1.11, precison: 83.58; recall: 84.21; f1: 83.89
# 1.1 , precison: 82.59; recall: 85.45; f1: 84.00   83.72
# 1.09, precison: 82.65; recall: 86.03; f1: 84.31   84.02
# 1.08, precison: 81.99; recall: 86.25; f1: 84.07   84.42
# 1.075,precison: 81.43; recall: 86.79; f1: 84.02   83.99
# 1.07, precison: 82.21; recall: 85.94; f1: 84.04   84.27
# 1.06, precison: 82.85; recall: 85.10; f1: 83.96   84.51
# 1.05, precison: 82.18; recall: 86.34; f1: 84.21   83.74
# 1.04, precison: 82.37; recall: 85.63; f1: 83.97   83.85
# 1.03, precison: 82.97; recall: 86.03; f1: 84.47   84.16
# 1.025,precison: 82.44; recall: 86.25; f1: 84.30   84.20
# 1.02, precison: 82.65; recall: 85.59; f1: 84.09   84.32
# 1.01, precison: 83.55; recall: 84.70; f1: 84.12   84.64
# 1.0 , precison: 82.92; recall: 86.17; f1: 84.51   83.98
# 0.99, precison: 83.51; recall: 84.48; f1: 83.99   83.89
# 0.98, precison: 83.13; recall: 86.12; f1: 84.60   84.09
# 0.975,precison: 82.02; recall: 85.81; f1: 83.87   84.25
# 0.97, precison: 82.11; recall: 86.39; f1: 84.20   84.24
# 0.96, precison: 82.31; recall: 85.68; f1: 83.96   84.46
# 0.95, precison: 82.64; recall: 86.21; f1: 84.39   84.18
# 0.94, precison: 82.01; recall: 86.57; f1: 84.22   84.66
# 0.93, precison: 81.95; recall: 86.25; f1: 84.05   84.23
# 0.925,precison: 82.97; recall: 85.63; f1: 84.28
# 0.92, precison: 82.41; recall: 85.68; f1: 84.01   84.09
# 0.91, precison: 82.12; recall: 86.21; f1: 84.11   84.04
# 0.9 , precison: 81.77; recall: 87.01; f1: 84.31   
# 0.89, 84.52
# 0.88, 84.37
# 0.875,84.11
# 0.87, 84.37
# 0.86, 84.38
# 0.85, precison: 82.61; recall: 86.03; f1: 84.29
# 0.84, 84.09
# 0.83, 83.81
# 0.825,84.24
# 0.82, 83.69
# 0.81, 84.12
# 0.8 , precison: 82.25; recall: 85.77; f1: 83.97   85.04
# 0.79, 83.88
# 0.78, 84.48
# 0.775,84.38
# 0.77, 84.61
# 0.76, 83.93
# 0.75, precison: 81.85; recall: 85.63; f1: 83.70   83.64
# 0.74, 84.23
# 0.73, 83.97
# 0.725,84.21
# 0.7 , precison: 82.18; recall: 85.94; f1: 84.02


# 截断策略beta:
# "learning_rate": 0.00032,
# 1, precison: 81.42; recall: 86.57; f1: 83.92;
# 2, precison: 82.07; recall: 86.74; f1: 84.34;
# 3, precison: 81.26; recall: 87.59; f1: 84.31;
# 4, precison: 81.88; recall: 86.43; f1: 84.09;
# 5, precison: 82.58; recall: 85.81; f1: 84.16;
# 6, precison: 82.49; recall: 86.12; f1: 84.27;
# 7, precison: 81.14; recall: 86.70; f1: 83.83
