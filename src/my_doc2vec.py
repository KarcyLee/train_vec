#!/usr/bin/python3
#encoding=utf-8
import os
from my_word2vec import *
def get_files(corpus_path):
    if os.path.exists(corpus_path) == False :
        print("corpus files don't exist !pls check the file path")
        return None
    try :
        result = []
        for parent,dirnames,filenames in os.walk(corpus_path):
            for filename in filenames:
                #print("parent folder is:" + parent)
                full_path = os.path.join(parent,filename)
                #print("filename with full path:"+ full_path)
                result.append(full_path)
        return result
    except Exception as err:
        print(err)
    else:
        return

if __name__ == "__main__":
    stop_words_file = "../tools/stop_list_1893.txt"
    stop_words = get_stopWords(stop_words_file)
    logger.info("停用词数目" + str( len(stop_words) ) )

    dirname = r"D:\Data\Corpus\tc-corpus-answer\answer\C35-Law"
    file_list = get_files(dirname)

    
    try:
        logger.info("开始训练")
        model = train_save(document_list_file, '.\out\word2vec_model')
        logger.info("训练结束")
    except Exception:
        logger.exeception("训练失败！")

