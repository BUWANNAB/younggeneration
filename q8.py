# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 11:43:57 2025

@author: I is God
"""

import re
import jieba
import jieba.posseg as pseg

class TextProcessor:
    def __init__(self, stopwords_path='E:/APP/QQNT/stopword.txt'):
        """初始化文本处理器，加载停用词"""
        self.stopwords = self._load_stopwords(stopwords_path)
        
    def _load_stopwords(self, path):
        """从文件加载停用词表"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return set([line.strip() for line in f])
        except FileNotFoundError:
            print(f"警告: 停用词文件 {path} 未找到，将使用空停用词表")
            return set()
            
    def preprocess_text(self, text):
        """预处理单个文本: 分词、去停用词、去特殊字符"""
        # 去除特殊字符
        text = re.sub(r'[^\w\s]', '', text)
        # 分词
        words = jieba.cut(text)
        # 去停用词
        return [word for word in words if word not in self.stopwords and word.strip()]
    
    def preprocess_corpus(self, corpus):
        """预处理文档集"""
        return [self.preprocess_text(doc) for doc in corpus]
    
    def filter_words(self, corpus, min_freq=2, allowed_pos=None):
        """过滤低频词和指定词性的词"""
        if allowed_pos is None:
            allowed_pos = {'n', 'nr', 'ns', 'nt', 'nz', 'v', 'vd', 'vn', 'a', 'ad', 'an'}
        
        # 统计词频
        from collections import Counter
        word_freq = Counter()
        for doc in corpus:
            word_freq.update(doc)
        
        # 过滤低频词
        filtered_corpus = []
        for doc in corpus:
            filtered_doc = []
            for word in doc:
                if word_freq[word] >= min_freq:
                    # 检查词性
                    if allowed_pos:
                        pos = pseg.cut(word).__next__().flag[0]  # 获取词性首字母
                        if pos in allowed_pos:
                            filtered_doc.append(word)
                    else:
                        filtered_doc.append(word)
            filtered_corpus.append(filtered_doc)
        
        return filtered_corpus

def load_corpus(file_path):
    """从文件加载语料库"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到")
        return []

def main():
    # 加载新闻文本
    news_corpus = load_corpus('E:/APP/QQNT/csgnews.txt')
    
    # 初始化文本处理器
    processor = TextProcessor('E:/APP/QQNT/stopword.txt')
    
    # 文本预处理
    preprocessed_corpus = processor.preprocess_corpus(news_corpus)
    
    # 过滤低频词和无关词性
    filtered_corpus = processor.filter_words(preprocessed_corpus)
    
    # 保存预处理结果
    with open('preprocessed_news.txt', 'w', encoding='utf-8') as f:
        for doc in filtered_corpus:
            f.write(' '.join(doc) + '\n')
    
    print(f"预处理完成，已保存 {len(filtered_corpus)} 篇新闻到 preprocessed_news.txt")

if __name__ == "__main__":
    main()    
    
    
    
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfExtractor:
    def __init__(self):
        """初始化TF-IDF关键词提取器"""
        self.vectorizer = TfidfVectorizer()
        
    def fit(self, corpus):
        """训练TF-IDF模型"""
        # 将分词后的文本转回字符串
        corpus_text = [' '.join(doc) for doc in corpus]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus_text)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
    def extract_keywords(self, doc_idx, top_n=10):
        """为指定文档提取关键词"""
        if doc_idx >= len(self.tfidf_matrix.toarray()):
            raise IndexError(f"文档索引 {doc_idx} 超出范围")
        
        # 获取文档的TF-IDF向量
        tfidf_vector = self.tfidf_matrix[doc_idx].toarray()[0]
        
        # 获取关键词及其TF-IDF值
        keywords = []
        for i, score in enumerate(tfidf_vector):
            if score > 0:
                keywords.append((self.feature_names[i], score))
        
        # 按TF-IDF值排序并返回前top_n个关键词
        keywords.sort(key=lambda x: x[1], reverse=True)
        return keywords[:top_n]
    
    def extract_keywords_for_all(self, top_n=10):
        """为所有文档提取关键词"""
        all_keywords = []
        for i in range(len(self.tfidf_matrix.toarray())):
            keywords = self.extract_keywords(i, top_n)
            all_keywords.append(keywords)
        return all_keywords

def load_preprocessed_corpus(file_path):
    """加载预处理后的语料库"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip().split() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到")
        return []

def main():
    # 加载预处理后的新闻文本
    preprocessed_corpus = load_preprocessed_corpus('preprocessed_news.txt')
    
    # 初始化TF-IDF提取器
    extractor = TfidfExtractor()
    
    # 训练模型
    extractor.fit(preprocessed_corpus)
    
    # 提取所有文档的关键词
    all_keywords = extractor.extract_keywords_for_all(top_n=10)
    
    # 输出结果
    for i, keywords in enumerate(all_keywords):
        print(f"\n新闻 {i+1} 的TF-IDF关键词:")
        for word, score in keywords:
            print(f"{word}: {score:.4f}")
    
    # 保存结果
    with open('tfidf_keywords.txt', 'w', encoding='utf-8') as f:
        for i, keywords in enumerate(all_keywords):
            f.write(f"新闻 {i+1} 的关键词:\n")
            for word, score in keywords:
                f.write(f"{word}: {score:.4f}\n")
            f.write("\n")

if __name__ == "__main__":
    main()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
import numpy as np
from collections import defaultdict, Counter

class TextRank:
    def __init__(self, window_size=5, damping=0.85, max_iter=100, tol=1e-6):
        """初始化TextRank算法参数"""
        self.window_size = window_size
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
        
    def extract_keywords(self, doc, top_n=10):
        """从单个文档中提取关键词"""
        if not doc:
            return []
            
        # 构建图的节点（词）
        words = list(set(doc))
        word2id = {w: i for i, w in enumerate(words)}
        id2word = {i: w for i, w in enumerate(words)}
        graph_size = len(words)
        
        # 如果文档太短，直接返回词频最高的词
        if graph_size < 2:
            freq = Counter(doc)
            keywords = [(w, freq[w]) for w in words]
            return sorted(keywords, key=lambda x: x[1], reverse=True)[:top_n]
        
        # 构建邻接矩阵
        matrix = np.zeros((graph_size, graph_size))
        for i in range(len(doc) - self.window_size + 1):
            window = doc[i:i+self.window_size]
            for j in range(len(window)):
                for k in range(j+1, len(window)):
                    w1, w2 = window[j], window[k]
                    matrix[word2id[w1], word2id[w2]] = 1
                    matrix[word2id[w2], word2id[w1]] = 1
        
        # 计算每个节点的出度
        out_degree = np.sum(matrix, axis=1)
        
        # 初始化PageRank值
        scores = np.ones(graph_size) / graph_size
        
        # 迭代计算
        for _ in range(self.max_iter):
            prev_scores = scores.copy()
            for i in range(graph_size):
                summation = 0
                for j in range(graph_size):
                    if matrix[j, i] > 0:
                        summation += matrix[j, i] / out_degree[j] * scores[j]
                scores[i] = (1 - self.damping) + self.damping * summation
            
            # 检查收敛
            if np.sum(np.abs(prev_scores - scores)) < self.tol:
                break
        
        # 获取排名最高的关键词
        sorted_indices = np.argsort(-scores)
        keywords = [(id2word[i], scores[i]) for i in sorted_indices[:top_n]]
        return keywords
    
    def extract_keywords_for_corpus(self, corpus, top_n=10):
        """从文档集中提取关键词"""
        all_keywords = []
        for doc in corpus:
            keywords = self.extract_keywords(doc, top_n)
            all_keywords.append(keywords)
        return all_keywords

def load_preprocessed_corpus(file_path):
    """加载预处理后的语料库"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip().split() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到")
        return []

def main():
    # 加载预处理后的新闻文本
    preprocessed_corpus = load_preprocessed_corpus('preprocessed_news.txt')
    
    # 初始化TextRank提取器
    textrank = TextRank(window_size=5)
    
    # 提取所有文档的关键词
    all_keywords = textrank.extract_keywords_for_corpus(preprocessed_corpus, top_n=10)
    
    # 输出结果
    for i, keywords in enumerate(all_keywords):
        print(f"\n新闻 {i+1} 的TextRank关键词:")
        for word, score in keywords:
            print(f"{word}: {score:.4f}")
    
    # 保存结果
    with open('textrank_keywords.txt', 'w', encoding='utf-8') as f:
        for i, keywords in enumerate(all_keywords):
            f.write(f"新闻 {i+1} 的关键词:\n")
            for word, score in keywords:
                f.write(f"{word}: {score:.4f}\n")
            f.write("\n")

if __name__ == "__main__":
    main()    