# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 09:17:17 2025

@author: I is God
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import spacy
from collections import Counter
import matplotlib.pyplot as plt

# 下载必要的NLTK数据
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')

# 词性标注函数
def pos_tagging(text):
    # 分词
    tokens = word_tokenize(text)
    # 词性标注
    tagged_tokens = pos_tag(tokens)
    return tagged_tokens

# 命名实体识别函数 - 使用NLTK
def ner_nltk(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    entities = nltk.chunk.ne_chunk(tagged)
    return entities

# 命名实体识别函数 - 使用spaCy (更精确)
def ner_spacy(text, nlp_model):
    doc = nlp_model(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# 可视化词性分布
def visualize_pos_distribution(tagged_tokens):
    pos_counts = Counter(tag for word, tag in tagged_tokens)
    top_pos = pos_counts.most_common(10)
    
    labels, values = zip(*top_pos)
    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.title('词性分布')
    plt.xlabel('词性标签')
    plt.ylabel('出现次数')
    plt.show()

# 可视化命名实体分布
def visualize_ner_distribution(entities):
    ent_types = [ent[1] for ent in entities]
    ent_counts = Counter(ent_types)
    
    labels, values = zip(*ent_counts.items())
    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.title('命名实体类型分布')
    plt.xlabel('实体类型')
    plt.ylabel('出现次数')
    plt.show()

def main():
    # 下载NLTK数据
    download_nltk_data()
    
    # 加载spaCy模型
    print("正在加载spaCy模型...")
    nlp = spacy.load("en_core_web_sm")
    
    # 示例文本 - 你可以替换为自己选择的文章
    sample_text = """
    Apple is looking at buying U.K. startup for $1 billion
    Apple Inc. is exploring a potential acquisition of a British technology startup, 
    according to sources familiar with the matter. The deal could value the company 
    at approximately $1 billion, marking Apple's largest acquisition since 2014.
    """
    
    print("\n=== 原始文本 ===")
    print(sample_text)
    
    # 词性标注
    print("\n=== 词性标注结果 ===")
    tagged = pos_tagging(sample_text)
    for word, tag in tagged[:20]:  # 只显示前20个结果
        print(f"{word}: {tag}")
    
    # 可视化词性分布
    visualize_pos_distribution(tagged)
    
    # NLTK命名实体识别
    print("\n=== NLTK命名实体识别结果 ===")
    entities_nltk = ner_nltk(sample_text)
    print(entities_nltk[:10])  # 只显示前10个结果
    
    # spaCy命名实体识别
    print("\n=== spaCy命名实体识别结果 ===")
    entities_spacy = ner_spacy(sample_text, nlp)
    for ent_text, ent_type in entities_spacy:
        print(f"{ent_text}: {ent_type}")
    
    # 可视化命名实体分布
    visualize_ner_distribution(entities_spacy)

if __name__ == "__main__":
    main()