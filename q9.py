# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 16:33:30 2025

@author: I is God
"""

import nltk
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split

class CorpusProcess:
    def __init__(self):
        self.sentences = []
    
    def process_sentence(self, sentence):
        # 简单的预处理，实际应用中可能需要更复杂的处理
        words = nltk.word_tokenize(sentence)
        tagged_words = nltk.pos_tag(words)
        return tagged_words
    
    def add_sentence(self, sentence):
        self.sentences.append(self.process_sentence(sentence))

class CRF_NER:
    def __init__(self):
        self.crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
    
    def word2features(self, sent, i):
        word = sent[i][0]
        postag = sent[i][1]
        
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],
        }
        
        if i > 0:
            word1 = sent[i-1][0]
            postag1 = sent[i-1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True
            
        if i < len(sent)-1:
            word1 = sent[i+1][0]
            postag1 = sent[i+1][1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True
                
        return features
    
    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]
    
    def sent2labels(self, sent):
        return [label for token, label in sent]
    
    def train(self, X_train, y_train):
        self.crf.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.crf.predict(X_test)

# 示例用法
if __name__ == "__main__":
    # 示例句子
    sentence = "2020年9月23日，'1+X'证书制度试点第四批职业教育培训评价组织和职业技能等级证书公示，其中广东泰迪智能科技股份有限公司申请的大数据应用开发（Python）位列其中。"
    
    # 初始化语料处理器
    corpus_processor = CorpusProcess()
    corpus_processor.add_sentence(sentence)
    
    # 准备训练数据（实际应用中需要真实的标注数据）
    # 这里仅作示例，使用模拟数据
    X = [corpus_processor.sent2features(s) for s in corpus_processor.sentences]
    y = [corpus_processor.sent2labels(s) for s in corpus_processor.sentences]
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 初始化并训练模型
    crf_ner = CRF_NER()
    crf_ner.train(X_train, y_train)
    
    # 预测
    y_pred = crf_ner.predict(X_test)
    
    # 输出命名实体（实际应用中需要根据标签解析）
    print("命名实体识别结果:", y_pred)