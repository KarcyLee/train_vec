#!/usr/bin/python3
#encoding=utf-8


from gensim.models import Word2Vec
import jieba
import codecs
import re
import multiprocessing
from Logger import *
log_file = "../log/my_word2vec.log"

logger = Logger(logname=log_file, loglevel=1, logger="word2vec").getlog()

def get_stopWords(stopWords_fn):
    '''
    获取停用词
    :param stopWords_fn: 停用词文件名
    :return: set,停用词
    '''
    try:
        stopWords_set = set()
        f = codecs.open(stopWords_fn, 'r','utf-8')
        lines = f.readlines()
        for line in lines:
            line = line.strip().rstrip()
            stopWords_set.add(line)
        f.close()
        return stopWords_set
    except Exception as e:
        logger.exception("获取停用词错误！")

def sentence2words(sentence, stopWords=False, stopWords_set=None):
    """
     split a sentence into words based on jieba 结巴分词
    """

    # seg_words is a generator
    seg_words = jieba.cut(sentence)
    if stopWords:
        words = [word for word in seg_words if word not in stopWords_set and word != ' ']
    else:
        words = [word for word in seg_words]
    return words

class MySentences(object):
    """
    文档分词器。不必一次都读入内存
    """
    def __init__(self, document_list_file):
        try :
            stopWords_fn = "../tools/stop_list_1893.txt"
            self.stopWords_set = get_stopWords(stopWords_fn)
            #self.pattern = re.compile(u'<content>(.*?)</content>')
            f = codecs.open(document_list_file, 'r','utf-8')
            lines = f.readlines()
            self.fns = []
            for line in lines:
                line_split = line.split(" ")
                self.fns.append(line_split[0])
            f.close()
        except Exception :
            logger.exception("构建文档分词器_构造函数错误！")

    def __iter__(self):

        for fn in self.fns:
            try:
                with codecs.open(fn, 'r','utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        if len(line) != 0:
                                yield sentence2words(line.strip(), True, self.stopWords_set)
                    '''
                    for line in f:
                        if line.startswith('<content>'):
                            content = self.pattern.findall(line)
                            if len(content) != 0:
                                yield sentence2words(content[0].strip(), True, self.stopWords_set)
                     '''
            except Exception:
                logger.error(fn)
                logger.exception("构建文档分词器_迭代器错误")

def train_save(document_list_file, model_filename):
    """
    训练并保存word_vec模型
    :param list_csv:
    :param model_fn:
    :return:
    默认sg=1是skip-gram算法，对低频词敏感；sg=0,CBOW算法，速度快，算近义词；
    size是向量的维数
    window是前后看词的单位，3表示在目标词前看3-b个词，后面看b个词（b在0-3之间随机）。
    negative和sample根据训练结果微调即可，sample采样虽然根据官网介绍设置1e-5
    """
    sentences = MySentences(document_list_file)
    num_features = 200
    min_word_count = 10
    num_workers =  multiprocessing.cpu_count()
    context = 5
    epoch = 20
    sample = 1e-5
    negative=3
    model = None
    try:
        model = Word2Vec(
        sentences,
        sg=0,
        size=num_features,
        min_count=min_word_count,
        workers=num_workers,
        negative=negative,
        sample=sample,
        window=context,
        #iter=epoch,
        )
        model.save(model_filename)
    except Exception:
        logger.exception("训练失败！")
    return model

if __name__ == "__main__":
    stop_words_file = "../tools/stop_list_1893.txt"
    stop_words = get_stopWords(stop_words_file)
    logger.info("停用词数目" + str( len(stop_words) ) )

    document_list_file = r"D:\Data\Corpus\all_samples.txt"
    #a = MySentences(document_list_file)
    try:
        logger.info("开始训练")
        model = train_save(document_list_file, '.\out\word2vec_model')
        logger.info("训练结束")
    except Exception:
        logger.exeception("训练失败！")


    # get the word vector
    for w in model.most_similar(u'互联网'):
        print( w[0], w[1])

    print( model.syn0.shape)

    print ( model.similarity(u'网络', u'互联网') )

    country_vec = model[u"国家"]
    print (type(country_vec))
    print (country_vec)

