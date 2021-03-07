import random
import numpy as np
import os
import re
import torch


class DataLoader(object):
    def __init__(self, data_dir, embedding_file, word_emb_dim, compute_device, seed=0, max_len=100, pos_dis_limit=50, other_label='Other'):
        self.data_dir = data_dir  # data root directory
        self.embedding_file = embedding_file  # embedding file path
        self.max_len = max_len  # specified max length of the sentence
        self.word_emb_dim = word_emb_dim
        self.seed = seed
        self.limit = pos_dis_limit  # limit of relative distance between word and entity
        # the label of other relations that do not consider to be evaluated
        self.other_label = other_label
        self.compute_device = compute_device

        self.word2idx = dict()
        self.label2idx = dict()
        self.label_nums = list()

        self.embedding_vectors = list()  # the index is consistent with word2idx
        self.unique_words = list()  # unique words of datasets

        self.original_words_num = 0
        self.lowercase_words_num = 0
        self.zero_digits_replaced_num = 0
        self.zero_digits_replaced_lowercase_num = 0

        self.word2idx['<pad>'] = self.pad_idx = len(
            self.word2idx)  # PAD character
        self.word2idx['<unk>'] = self.unk_idx = len(
            self.word2idx)  # out of vocabulary
        self.word2idx['<e1>'] = len(self.word2idx)
        self.word2idx['<e2>'] = len(self.word2idx)
        self.word2idx['</e1>'] = len(self.word2idx)
        self.word2idx['</e2>'] = len(self.word2idx)

        # load unique words
        vocab_path = os.path.join(self.data_dir, 'words.txt')
        with open(vocab_path, 'r') as f:
            for line in f:
                self.unique_words.append(line.strip())

        # load labels (labels to indices)
        labels_path = os.path.join(data_dir, 'labels.txt')
        with open(labels_path, 'r') as f:
            for i, line in enumerate(f):
                lable, label_num = line.strip().split()
                self.label2idx[lable] = i
                self.label_nums.append(int(label_num))

        # get the relation labels to be evaluated
        other_label_idx = self.label2idx[self.other_label]
        self.metric_labels = list(self.label2idx.values())
        self.metric_labels.remove(other_label_idx)

    def load_embeddings_from_file_and_unique_words(self, verbose=True):
        embedding_words = [emb_word for emb_word, _ in self.load_embeddings_from_file(
            emb_path=self.embedding_file)]
        # emb_word <-> [unique_word_1, unique_word_2, ...]
        emb_word2unique_word = dict()
        out_of_vocab_words = list()  # 不在预训练的词汇中
        word_vecs = list()
        for unique_word in self.unique_words:
            emb_word = self.get_embedding_word(unique_word, embedding_words)
            if emb_word is None:  # 说明这个词的各种形式都不在预训练的词表中
                out_of_vocab_words.append(unique_word)
            else:
                if emb_word not in emb_word2unique_word:
                    emb_word2unique_word[emb_word] = [
                        unique_word]  # 注意这里的list形式，方便下面append（）
                else:
                    # emb_word是忽略大小写的词，这里是把词的各种形式放到统一的键下
                    emb_word2unique_word[emb_word].append(unique_word)

        for emb_word, emb_vector in self.load_embeddings_from_file(emb_path=self.embedding_file):
            if emb_word in emb_word2unique_word:
                for unique_word in emb_word2unique_word[emb_word]:
                    self.word2idx[unique_word] = len(self.word2idx)
                    word_vecs.append(emb_vector)  # 只有大小写区别的词使用同一词向量

        word_vecs = np.stack(word_vecs)
        vec_mean, vec_std = word_vecs.mean(), word_vecs.std()
        special_emb = np.random.normal(
            vec_mean, vec_std, (6, self.word_emb_dim))
        special_emb[0] = 0  # <pad> is initialize as zero

        word_vecs = np.concatenate((special_emb, word_vecs), axis=0)
        word_vecs = word_vecs.astype(np.float32).reshape(-1, self.word_emb_dim)
        self.embedding_vectors = torch.from_numpy(word_vecs)

        if verbose:
            print('\nloading vocabulary from embedding file and unique words:')
            print('    First 20 OOV words:')
            for i, oov_word in enumerate(out_of_vocab_words):
                print('        out_of_vocab_words[%d] = %s' % (i, oov_word))
                if i > 20:
                    break
            print(' -- len(out_of_vocab_words) = %d' % len(out_of_vocab_words))
            print(' -- original_words_num = %d' % self.original_words_num)
            print(' -- lowercase_words_num = %d' % self.lowercase_words_num)
            print(' -- zero_digits_replaced_num = %d' %
                  self.zero_digits_replaced_num)
            print(' -- zero_digits_replaced_lowercase_num = %d' %
                  self.zero_digits_replaced_lowercase_num)

    def load_embeddings_from_file(self, emb_path):
        """Load word embedding from file"""
        with open(emb_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                if len(values) != self.word_emb_dim + 1:
                    continue
                emb_vector = np.asarray(values[1:], dtype=np.float32)

                yield word, emb_vector

    def get_embedding_word(self, word, embedding_words):
        """Mapping of words in datsets into embedding words"""
        if word in embedding_words:
            self.original_words_num += 1
            return word
        elif word.lower() in embedding_words:
            self.lowercase_words_num += 1
            return word.lower()
        elif re.sub(r'\d', '0', word) in embedding_words:
            self.zero_digits_replaced_num += 1
            return re.sub(r'\d', '0', word)
        elif re.sub(r'\d', '0', word.lower()) in embedding_words:
            self.zero_digits_replaced_lowercase_num += 1
            return re.sub(r'\d', '0', word.lower())
        return None

    def load_sentences_labels(self, sentences_file, labels_file, use_pi, d):
        """Loads sentences and labels from their corresponding files. 
            Maps tokens and tags to their indices and stores them in the provided dict d.
        """
        sents, pos1s, pos2s, entities, slices = list(), list(), list(), list(), list()
        labels = list()

        # Replace each token by its index if it is in vocab, else use index of unk_word
        with open(sentences_file, 'r') as f:
            for i, line in enumerate(f):
                e1, e2, sent = line.strip().split('\t')
                words = sent.split(' ')
                words = words[:self.max_len] if len(
                    words) > self.max_len else words
                e1_start = e1.split(' ')[0] if ' ' in e1 else e1
                e2_start = e2.split(' ')[0] if ' ' in e2 else e2
                # e1 index in the words/sent
                e1_start_idx = words.index(e1_start)
                # e2 index in the words/sent
                e2_start_idx = words.index(e2_start)
                if use_pi:
                    e1_end = e1.split(' ')[-1] if ' ' in e1 else e1
                    e2_end = e2.split(' ')[-1] if ' ' in e2 else e2
                    # e1 index in the words/sent
                    e1_end_idx = words.index(e1_end)
                    # e2 index in the words/sent
                    e2_end_idx = words.index(e2_end)
                    words.insert(e1_start_idx, '<e1>')
                    words.insert(e1_end_idx+2, '</e1>')
                    words.insert(e2_start_idx+2, '<e2>')
                    words.insert(e2_end_idx+4, '</e2>')
                sent, pos1, pos2 = list(), list(), list()
                for idx, word in enumerate(words):
                    emb_word = self.get_embedding_word(word, self.word2idx)
                    if emb_word:
                        sent.append(self.word2idx[word])
                    else:
                        sent.append(self.unk_idx)
                    pos1.append(self.get_pos_feature(idx - e1_start_idx))
                    pos2.append(self.get_pos_feature(idx - e2_start_idx))
                sents.append(sent)
                pos1s.append(pos1)
                pos2s.append(pos2)

                if use_pi:
                    e1_idx = [i for i in range(sent.index(self.word2idx['<e1>'])+1,
                                               sent.index(self.word2idx['</e1>']))]
                    e2_idx = [i for i in range(sent.index(self.word2idx['<e2>'])+1,
                                               sent.index(self.word2idx['</e2>']))]
                    entity_idx = e1_idx+e2_idx
                    entities.append(entity_idx)

                    slice1 = [i for i in range(0, sent.index(self.word2idx['<e1>'])+1)]
                    slice2 = [i for i in range(sent.index(self.word2idx['</e1>']),
                                               sent.index(self.word2idx['<e2>'])+1)]
                    slice3 = [i for i in range(sent.index(self.word2idx['</e2>']),
                                               len(sent))]
                    slice = slice1+slice2+slice3
                    slices.append(slice)

        # Replace each label by its index
        with open(labels_file, 'r') as f:
            for line in f:
                idx = self.label2idx[line.strip()]
                labels.append(idx)

        # Check to ensure there is a tag for each sentence
        assert len(labels) == len(sents)

        # Storing data and labels in dict d
        d['data'] = {'sents': sents, 'pos1s': pos1s, 'pos2s': pos2s, 'entities': entities, 'slices':slices}
        d['labels'] = labels
        d['size'] = len(sents)

    def get_pos_feature(self, x):
        """Clip the relative postion range:
            -limit ~ limit => 0 ~ limit * 2+2
        """
        if x < -self.limit:
            return 0
        elif x >= -self.limit and x <= self.limit:
            return x + self.limit + 1
        else:
            return self.limit * 2 + 2

    def load_data(self, data_type, use_pi=False):
        """Loads the data for each type in types from data_dir.

        Args:
            data_type: (str) has one of 'train', 'val', 'test' depending on which data is required.
        Returns:
            data: (dict) contains the data with tags for each type in types.
        """
        data = dict()

        if data_type in ['train', 'val', 'test']:
            sentences_file = os.path.join(
                self.data_dir, data_type, 'sentences.txt')
            labels_file = os.path.join(self.data_dir, data_type, 'labels.txt')
            self.load_sentences_labels(
                sentences_file, labels_file, use_pi, data)
        else:
            raise ValueError("data type not in ['train', 'val', 'test']")
        return data

    def data_iterator(self, data, batch_size, shuffle='False', use_cnn=False):
        """Returns a generator that yields batches data with tags.
        Args:
            data: (dict) contains data which has keys 'data', 'labels' and 'size'
            batch_size: (int) batch size
            shuffle: (bool) whether the data should be shuffled

        Yields:
            batch_data: (tensor) shape: batch_size x max_seq_len
            batch_tags: (tensor) shape: batch_size x max_seq_len
        """
        order = list(range(data['size']))
        if shuffle:
            random.seed(self.seed)  # 保证结果可以复现
            random.shuffle(order)
        # one pass over data
        for i in range((data['size']) // batch_size):
            # fetch data and labels
            batch_sents = [data['data']['sents'][idx]
                           for idx in order[i * batch_size:(i + 1) * batch_size]]
            batch_pos1s = [data['data']['pos1s'][idx]
                           for idx in order[i * batch_size:(i + 1) * batch_size]]
            batch_pos2s = [data['data']['pos2s'][idx]
                           for idx in order[i * batch_size:(i + 1) * batch_size]]
            batch_entities = [data['data']['entities'][idx]
                           for idx in order[i * batch_size:(i + 1) * batch_size]]
            batch_slices = [data['data']['slices'][idx]
                              for idx in order[i * batch_size:(i + 1) * batch_size]]
            batch_labels = [data['labels'][idx]
                            for idx in order[i * batch_size:(i + 1) * batch_size]]

            # 供lstm使用
            batch_sent_lens = list(map(len, batch_sents))
            lens_tuple_list = [(idx, len)
                               for idx, len in enumerate(batch_sent_lens)]
            lens_tuple_list.sort(reverse=True, key=lambda t: t[1])
            indices, sorted_batch_lens = [t[0] for t in lens_tuple_list], [
                t[1] for t in lens_tuple_list]
            if use_cnn:
                batch_max_len = self.max_len
            else:
                batch_max_len = max(batch_sent_lens)
            slice_lens = list(map(len, batch_slices))
            batch_max_slice_len = max(slice_lens)
            entity_lens = list(map(len, batch_entities))
            batch_max_entity_len = max(entity_lens)

            """分为词特征和位置特征，词特征填充pad，位置特征填self.limit*2+2"""
            batch_data_sents = self.pad_idx * \
                np.ones((batch_size, batch_max_len))
            batch_data_pos1s = (self.limit * 2 + 2) * \
                np.ones((batch_size, batch_max_len))
            batch_data_pos2s = (self.limit * 2 + 2) * \
                np.ones((batch_size, batch_max_len))
            batch_data_entities = (batch_max_len-1) * np.ones((batch_size, batch_max_entity_len))
            batch_data_entity_masks = torch.zeros((batch_size, batch_max_entity_len)).bool()
            batch_data_slice_masks = torch.zeros((batch_size, batch_max_slice_len)).bool()
            batch_data_slices = (batch_max_len-1) * np.ones((batch_size, batch_max_slice_len))
            batch_data_mask = torch.zeros((batch_size, batch_max_len)).bool()

            if use_cnn:         # cnn是保持长度固定，lstm选用批最大长度
                for j in range(batch_size):
                    cur_len = len(batch_sents[j])
                    min_len = min(cur_len, batch_max_len)
                    batch_data_sents[j][:min_len] = batch_sents[j][:min_len]
                    batch_data_pos1s[j][:min_len] = batch_pos1s[j][:min_len]
                    batch_data_pos2s[j][:min_len] = batch_pos2s[j][:min_len]
                    batch_data_mask[j][:min_len] = torch.Tensor([True] * int(min_len))
            else:
                for j, (sent, pos1, pos2, sent_len, slice_len, slice, entity_len, entity) in enumerate(
                        zip(batch_sents, batch_pos1s, batch_pos2s, batch_sent_lens,
                            slice_lens, batch_slices, entity_lens, batch_entities)):
                    batch_data_sents[j][:sent_len] = sent
                    batch_data_pos1s[j][:sent_len] = pos1
                    batch_data_pos2s[j][:sent_len] = pos2
                    batch_data_mask[j][:sent_len] = torch.Tensor([True] * int(sent_len))
                    batch_data_entities[j][:entity_len] = entity
                    batch_data_entity_masks[j][:entity_len] = torch.Tensor([True] * int(entity_len))
                    batch_data_slices[j][:slice_len] = slice
                    batch_data_slice_masks[j][:slice_len] = torch.Tensor([True] * int(slice_len))

            # Convert indices data to torch LongTensors, and shift tensors to GPU if available.
            batch_data_sents = torch.LongTensor(batch_data_sents)[
                indices].to(self.compute_device)
            batch_data_pos1s = torch.LongTensor(batch_data_pos1s)[
                indices].to(self.compute_device)
            batch_data_pos2s = torch.LongTensor(batch_data_pos2s)[
                indices].to(self.compute_device)
            batch_data_mask = torch.BoolTensor(batch_data_mask)[
                indices].to(self.compute_device)
            batch_data_entities = torch.LongTensor(
                batch_data_entities)[indices].to(self.compute_device)
            batch_data_slices = torch.LongTensor(
                batch_data_slices)[indices].to(self.compute_device)
            batch_labels = torch.LongTensor(
                batch_labels)[indices].to(self.compute_device)
            batch_slice_masks = torch.BoolTensor(
                batch_data_slice_masks)[indices].to(self.compute_device)
            batch_slice_lens = torch.LongTensor(slice_lens)[indices].to(self.compute_device)
            batch_entity_masks = torch.BoolTensor(
                batch_data_entity_masks)[indices].to(self.compute_device)

            batch_data = {'sents': batch_data_sents, 'pos1s': batch_data_pos1s, 'pos2s': batch_data_pos2s,
                          'entities': batch_data_entities, 'entity_masks': batch_entity_masks,
                          'slices':batch_data_slices, 'slice_masks': batch_slice_masks, 'slice_lens':batch_slice_lens,
                          'mask': batch_data_mask, 'lens': sorted_batch_lens}

            yield batch_data, batch_labels
