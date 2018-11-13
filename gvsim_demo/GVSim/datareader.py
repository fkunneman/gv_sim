
import pickle
from collections import defaultdict
import re

from pymongo import MongoClient

import numpy
import string
punctuation = string.punctuation
import time
from gensim.summarization import bm25
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from sklearn.metrics.pairwise import cosine_similarity

class Datareader:

    def __init__(self, dbport, dbname, qcl, acl, qa):

        client = MongoClient('localhost', dbport)
        self.db = client[dbname]
        
        # loading stopwords
        STOPWORDS_PATH='/var/lib/goeievraag/stopwords.txt'
        with open(STOPWORDS_PATH) as f:
            self.stopwords = [word.lower().strip() for word in f.read().split()]
        
        # read questions
        self.qcl = self.db[qcl]
        self.fast_q = {}
        for q in self.qcl.find({}):
            self.fast_q[q['i']] = q['text']

        # read answers
        self.acl = self.db[acl]
        self.fast_a = {}
        for a in self.acl.find({}):
            self.fast_a[a['i']] = a['text']
        
        # read question_answers dictionary
        self.qa = self.db[qa]
        self.fast_qa = {}
        for q in self.qa.find({}):
            self.fast_qa[q['qid']] = q['aids']

    def retrieve_by_list(self,coll,k,l):
        return list(coll.aggregate([{'$match' : {k: {'$in':l}}}]))
        
    def return_questions_answerids(self,indices):
        answers = []
        for index in indices:
            answers.append(self.fast_qa[index][0])
        answers_txt = [self.fast_a[i] for i in answers]
        return answers_txt
        
        # answers_db = self.retrieve_by_list(self.acl,'i',answers)
        # answers_sorted = []
        # for a in answers:
        #     for adb in answers_db:
        #         if adb['i'] == a:
        #             answers_sorted.append(adb['text'])
        #             break
        # return answers_sorted

    def init_bm25(self, bm25_path):
        # read bm25 model
        with open(bm25_path, 'rb') as fid:
            self.bm25 = pickle.load(fid)
            self.avg_idf = sum(map(lambda k: float(self.bm25.idf[k]), self.bm25.idf.keys())) / len(self.bm25.idf.keys())

    def init_soft_cosine(self, dict_path, tfidf_path):
        # read dict
        self.dict = Dictionary.load(dict_path)
        # read tfidf
        self.tfidf = TfidfModel.load(tfidf_path)

    def init_word2vec(self, word2vec_path):
        self.word2vec = Word2Vec.load(word2vec_path)
        
    def preprocess(self, query):
        tokens = re.sub(r'([.,;:?!\'\(\)-])', r' \1 ', query.lower()).split()
        # remove stopwords and punctuation
        tokens = [w for w in tokens if w not in self.stopwords and w not in punctuation]
        return tokens

    def encode(self, question):
        emb = []
        for w in question:
            try:
                emb.append(self.word2vec[w.lower()])
            except:
                emb.append(300 * [0])
        return emb
    
    def softcos(self, q1, q2):

        def dot(q1tfidf, q1emb, q2tfidf, q2emb):
            cos = 0.0
            for i, w1 in enumerate(q1tfidf):
                for j, w2 in enumerate(q2tfidf):
                    m_ij = max(0, cosine_similarity([q1emb[i]], [q2emb[j]])[0][0])**2
                    cos += (w1[1] * m_ij * w2[1])
            return cos

        q1emb = self.encode(q1)
        q2emb = self.encode(q2)

        q1tfidf = self.tfidf[self.dict.doc2bow(q1)]
        q2tfidf = self.tfidf[self.dict.doc2bow(q2)]

        q1q1 = numpy.sqrt(dot(q1tfidf, q1emb, q1tfidf, q1emb))
        q2q2 = numpy.sqrt(dot(q2tfidf, q2emb, q2tfidf, q2emb))
        softcosine = dot(q1tfidf, q1emb, q2tfidf, q2emb) / (q1q1 * q2q2)
        return softcosine
    
    def retrieve(self, query, n=30):

        # retrieve scores from model
        scores = self.bm25.get_scores(query, self.avg_idf)
        scores_numbers = [[j,score] for j,score in enumerate(scores)]
        scores_numbers_ranked = sorted(scores_numbers,key = lambda k : k[1],reverse=True)
        # question_indices, questions = [x[0] for x in scores_numbers_ranked[:n]], [self.qcl.find_one({'i':q[0]})['text'] for q in scores_numbers_ranked[:n]]
        question_indices, questions = [x[0] for x in scores_numbers_ranked[:n]], [self.fast_q[q[0]] for q in scores_numbers_ranked[:n]]

        return question_indices, questions

    def rerank(self, query, question_indices, questions, questions_tok, n=5):
        output = []
        for i, question in enumerate(questions_tok):
            output.append([question_indices[i], questions[i], question, self.softcos(query, question)])

        questions = sorted(output, key=lambda x: x[3], reverse=True)[:n]
        return questions
    
    def run(self, query):
        start = time.time()
        # preprocess query
        query_tok = self.preprocess(query)

        # retrieve 30 candidates with bm25
        question_indices, questions = self.retrieve(query_tok)
        # preprocess candidates
        questions_tok = [self.preprocess(q) for q in questions]

        # reranking with softcosine
        questions_ranked = self.rerank(query_tok, question_indices, questions, questions_tok)

        # retrieve answers and prepare question objects
        answers_ranked = self.return_questions_answerids([q[0] for q in questions_ranked])
        answers_ranked_splitted = []
        for answer in answers_ranked:
            answer_splitted = answer.split('\n')
            answers_ranked_splitted.append(answer_splitted)
        question_objects = [{'question':q[1], 'answer':answers_ranked_splitted[i], 'score': q[3]} for i,q in enumerate(questions_ranked)]
        end = time.time()
        
        return question_objects, round(end-start, 2)
