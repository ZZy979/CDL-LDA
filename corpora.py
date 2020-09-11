import os
import pickle

import numpy as np
from gensim import interfaces


class Document:
    """表示语料库中的一篇文档"""

    def __init__(self, words, label, domain):
        """构造一篇文档

        :param words: list of int，单词列表，每个元素表示单词索引
        :param label: int，文档标签（主题组）
        :param domain: int，文档所属的域，0 - 源域，1 - 目标域
        """
        self.words = words
        self.label = label
        self.domain = domain

    def __iter__(self):
        return iter(self.words)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        return self.words[index]


class Corpus(interfaces.CorpusABC):
    """表示一个数据集的语料库，可迭代产生 :class:`Document` 对象。

    属性
    =====
    * name: 数据集名称
    * id2word：字典{id: word}，表示单词表
    * n_vocabs：单词表大小
    * n_labels：标签数

    方法
    =====
    * for doc in corpus：迭代产生 :class:`Document` 对象doc
    * len(corpus)：返回文档数
    """

    def __init__(self, name, base_path):
        self.name = name
        with open(os.path.join(base_path, name, 'data.pkl'), 'rb') as f:
            docs = pickle.load(f)
            self.word2id = pickle.load(f)
            self.id2word = pickle.load(f)
            labels = pickle.load(f)
            domains = pickle.load(f)

            self.n_vocabs = len(self.id2word)
            self.n_labels = len(set(labels))
            self.docs = [
                Document(docs[i], labels[i], domains[i])
                for i in range(len(docs)) if len(docs[i]) >= 5
            ]

    def __iter__(self):
        return iter(self.docs)

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, d):
        return self.docs[d]

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, metadata=False):
        pass


class PriorCorpus(Corpus):
    """带有单词标签先验的语料库。

    增加的属性
    =====
    * prior: ndarray(V, L)，表示单词的标签先验概率
    """

    def __init__(self, name, base_path):
        super().__init__(name, base_path)
        with open(os.path.join(base_path, name, 'soft_prior.pkl'), 'rb') as f:
            self.prior = np.array(pickle.load(f))
