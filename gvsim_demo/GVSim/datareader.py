
import pickle
from collections import defaultdict
import numpy
import re

class Datareader:

    def __init__(self, questionpath, examplequestionpath, answerpath, question_answers, bm25_path):
        # read questions
        with open(questionpath,'r',encoding='utf-8') as file_in:
            self.questions = numpy.array(file_in.read().strip().split('\n'))

        # read example questions
        with open(examplequestionpath,'r',encoding='utf-8') as file_in:
            indices = [[int(x) for x in line.split()] for line in file_in.read().strip().split('\n')]
            self.examples_returns = defaultdict(list)
            for i in indices:
                self.examples_returns[i[0]].append(i[1])           
            
        # read answers
        with open(answers,'r',encoding='utf-8') as file_in:
            self.answers = numpy.array(file_in.read().strip().split('\n'))

        # read question_answers dictionary
        with open(question_answers) as file_in:
            self.qa = defaultdict(list)
            for q_a in file_in.read().strip().split('\n'):
                tokens = q_a.split()
                self.qa[int(tokens[0])].append(tokens[1])
                
        # read bm25 model
        with open(bm25_path, 'rb') as fid:
            self.bm25 = pickle.load(fid)
        self.avg_idf = sum(map(lambda k: float(self.bm25.idf[k]), self.bm25.idf.keys())) / len(self.bm25.idf.keys())
        
    def return_questions_answers(self,index):
        example_questions = []
        example_answers = []
        for i in self.example_question_indices:
            example_questions.append(self.questions[i])
            example_answers.append(self.answers[self.qa[i]])
        return example_questions, example_answers
            
    def rank_sims_bm25(self, query, similarity_threshold):
        # tokenize query
        query_tok = re.sub(r'([.,;:?!\'\(\)-])', r' \1 ', query)

        # retrieve scores from model
        scores = self.bm25.get_scores(query_tok, average_idf)
        scores_numbers = [[j,score] for j,score in enumerate(scores)]
        scores_numbers_ranked = sorted(scores_numbers,key = lambda k : k[1],reverse=True)
        top10_sim_qa = [[self.questions[x[0]],self.answers[self.qa[x[0]]]] for x in scores_numbers_ranked[:10] if x[1] > similarity_threshold]
        top5_nosim_qa = []
        for x in scores_numbers_ranked:
            if x[1] < similarity_threshold:
                top5_nosim_qa.append([self.questions[x[0]],[self.answers[i] for i in self.qa[x[0]]]])
            if len(top5_nosim_qa) >= 5:
                break
        return top10_sim_qa, top5_nosim_qa
