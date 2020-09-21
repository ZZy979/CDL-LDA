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


class CrossDomainCorpus(interfaces.CorpusABC):
    """表示一个跨域语料库，包含源域和目标域的文档，前n_source个为源域文档，之后为目标域文档。

    属性
    =====
    * name: 数据集名称
    * n_labels：标签数
    * n_source：源域文档数

    方法
    =====
    * for doc in corpus：迭代产生 :class:`Document` 对象doc
    * len(corpus)：总文档数
    * corpus[d]：返回编号为d的文档
    """

    def __init__(self, name, docs, labels, domains):
        """构造一个跨域语料库。

        :param name: 数据集名称
        :param docs: list of list of int，每个列表为一篇文档，每个元素为单词在单词表中的索引
        :param labels: list of int，文档的标签，每个元素为标签的编号(0~L-1)
        :param domains: list of int，文档的域（已排序），0 - 源域，1 - 目标域
        """
        self.name = name
        self.docs = [
            Document(docs[i], labels[i], domains[i])
            for i in range(len(docs)) if len(docs[i]) >= 5
        ]
        self.n_labels = len(set(labels))
        self.n_source = sum(1 for doc in self.docs if doc.domain == 0)

    def __iter__(self):
        return iter(self.docs)

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, d):
        return self.docs[d]

    @staticmethod
    def save_corpus(fname, corpus, id2word=None, metadata=False):
        pass
