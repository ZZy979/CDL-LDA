from gensim import interfaces


class Document:
    """表示语料库中的一篇文档"""

    def __init__(self, words, label, domain):
        """构造一篇文档

        :param words: list of int，单词列表，每个元素表示单词索引
        :param label: int，文档标签
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
    * n_labels：标签数

    方法
    =====
    * for doc in corpus：迭代产生 :class:`Document` 对象doc
    * len(corpus)：返回文档数
    """

    def __init__(self, name, docs, labels, domains):
        self.name = name
        self.docs = [
            Document(docs[i], labels[i], domains[i])
            for i in range(len(docs)) if len(docs[i]) >= 5
        ]
        self.n_labels = len(set(labels))

    def __iter__(self):
        return iter(self.docs)

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, d):
        return self.docs[d]

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, metadata=False):
        pass
