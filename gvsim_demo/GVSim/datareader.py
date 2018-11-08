
import pickle
from collections import defaultdict
import numpy
import re

from pymongo import MongoClient

class Datareader:

    def __init__(self, dbport, dbname, qcl, eqcl, acl, qa):

        client = MongoClient('localhost', dbport)
        self.db = client[dbname]
        
        # read questions
        self.qcl = self.db[qcl]

        # read example questions
        self.eqcl = self.db[eqcl]
        self.examples = [q['text'] for q in self.eqcl.find({})]

        # read answers
        self.acl = self.db[acl]

        # read question_answers dictionary
        self.qa = self.db[qa]

    def retrieve_by_list(self,coll,k,l):
        return list(coll.aggregate([{'$match' : {k: {'$in':l}}}]))
        
    def return_question_answers(self,index):
        db_entry = self.qa.find({'qid':index})
        answer_ids = db_entry[0]['aids']
        answers = [r['text'] for r in self.retrieve_by_list(self.acl,'i',answer_ids)]
        return answers

    def rank_sims_bm25(self, bm25_path, query, similarity_threshold):
        # read bm25 model
        with open(bm25_path, 'rb') as fid:
            self.bm25 = pickle.load(fid)
            self.avg_idf = sum(map(lambda k: float(self.bm25.idf[k]), self.bm25.idf.keys())) / len(self.bm25.idf.keys())

        # tokenize query
        query_tok = re.sub(r'([.,;:?!\'\(\)-])', r' \1 ', query.lower()).split()

        # retrieve scores from model
        scores = self.bm25.get_scores(query_tok, self.avg_idf)
        scores_numbers = [[j,score] for j,score in enumerate(scores)]
        scores_numbers_ranked = sorted(scores_numbers,key = lambda k : k[1],reverse=True)
        questions = self.retrieve_by_list(self.qcl,'i',[x[0] for x in scores_numbers_ranked[:5]]) 
        top5_sim_qa = [{'question':q['text'], 'answers':self.return_question_answers(q['i'])} for q in questions]
        top5_nosim_qa = []
        # for related questions (not part of demo yet)
#        for x in scores_numbers_ranked:
#            if x[1] < similarity_threshold:
#                top5_nosim_qa.append([self.questions[x[0]],[self.answers[i] for i in self.qa[x[0]]]])
#            if len(top5_nosim_qa) >= 5:
#                break
        return top5_sim_qa

    ##########################################
    ### PLACE SOFT COSINE MODEL HERE #########
    ##########################################
    def rank_sims_soft_cosine(self):
        pass

