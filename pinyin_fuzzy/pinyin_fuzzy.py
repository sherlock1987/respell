import math
import os
from re import A
import jieba
import pickle
import logging
from tqdm import tqdm
jieba.setLogLevel(log_level=logging.INFO)
import pandas as pd
from fuzzywuzzy import process
from xpinyin import Pinyin
from Pinyin2Hanzi import DefaultDagParams
from Pinyin2Hanzi import dag
from gensim import corpora, similarities, models

def hanzi2pinyin(ch):                      
        p=Pinyin()
        s=p.get_pinyin(u'%s'%(ch)).split('-')
        return s

    

def domain_in_query(root_path, data_name, b):
    """
    root_path: 根目录
    data_name: 领域的名字
    b: query
    """
    words=[]
    data = pd.read_csv(os.path.join(root_path, "pinyin_fuzzy/{}pinyin.txt".format(data_name)), encoding="utf-8",sep='\t')
    data_split_word = data.word.apply(jieba.lcut)
    dictionary = corpora.Dictionary(data_split_word.values)
    data_corpus = data_split_word.apply(dictionary.doc2bow)
    trantab = str.maketrans("0123456789", "0123456789")
    b=jieba.lcut(b)
    for i in range(len(b)-1):
            text=b[i]
            temp=b[i]
            if len(text)==1:
                text=b[i]+b[i+1]
            text=hanzi2pinyin(text)
            text=''.join(text)

            find_corpus = [dictionary.doc2bow(jieba.lcut(text.translate(trantab)))]

            tfidf = models.TfidfModel(data_corpus.to_list())
            index = similarities.SparseMatrixSimilarity(
                tfidf[data_corpus], num_features=len(dictionary))

            result = []
            for corpus in find_corpus:
                sim = pd.Series(index[corpus])
                result.append(data.word[sim.nlargest(1).index].index)

            word_=[]
            for i in range(len(result[0])):
                if sim.nlargest(1).values[i]>=1:
                    content =data.words[result[0].values[i]]
                    words.append(content)
    words=list(set(words))
    words=','.join(words)
    return words 

        
if __name__ == '__main__':
    data_name_list = ["law", "med", "odw"]
    for data_name in data_name_list:
        count_line = 0
        file_data=''
        root_path = "/Users/hzx/Desktop/work_project/python_project/respell"
        with open(os.path.join(root_path, "csc_evaluation/builds/sim/domain/{}.train".format(data_name)), 'r', encoding='utf-8') as f:
            datas=f.readlines()
            for i,data in enumerate(tqdm(datas)):
                # a: 错误的数字 b：错误的query c：正确的回答
                a,b,c=data.strip().split('\t')
                query_content=c

                result_domain = domain_in_query(root_path, data_name,b)
                if len(result_domain)!=0:
                    b=b+'，领域词是'+result_domain
                else:
                    b=b
                l=a+'\t'+b+'\t'+c+'\n'
                file_data+=l
                count_line += 1
                # if count_line == 3:
                #     break

            # 加入js代表了引入检索之后的结果，每一个for循环都写出最终结果
            with open(os.path.join(root_path, "csc_evaluation/builds/sim/domain/{}_retrieve_respell.train".format(data_name)), 'w', encoding='utf-8') as f:
                for i in file_data:
                    f.write(i)
