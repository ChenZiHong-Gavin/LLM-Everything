# Bag-of-words

## 1 bag-of-words

词袋模型（Bag-of-Words model，BOW）的基本思想是：假定对于一段文本，忽略其词序和语法，仅将其当作一个词汇的集合，如同一个袋子一样。

### 1.1 从one-hot讲起

例如一句话`I love you`，每个词都是一个数值表示：

```
I: [1, 0, 0]
love: [0, 1, 0]
you: [0, 0, 1]
```

这是一种简单的编码方式，但是存在着维度灾难（词越多越稀疏）和语义鸿沟（无法进行相似度计算）等问题。

### 1.2 tf-idf

如果仅仅考虑词汇的词频（即TF），会导致一些常见词汇的权重过高，而忽略了一些重要的词汇。

TF-IDF（Term Frequency-Inverse Document Frequency）通过引入逆文档频率（IDF）来更好地体现文本特征。TF-IDF的计算公式如下：

```
TF(t) = frequency of t in document / total words in document
IDF(t) = log(total documents / number of documents with term t in it + 1)
TF-IDF(t) = TF(t) * IDF(t)
```
