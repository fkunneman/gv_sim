
from django.shortcuts import render
from django.views.generic import View

from . import datareader

import sys

class GVHome(View):

    print('SET DATA AND MODELS')
    # for local testing
    datareader = datareader.Datareader(27020,'GoeieVraag','cqa','answers','qa_dict')
    datareader.init_bm25('/roaming/fkunnema/goeievraag/exp_similarity_new/bm25.pkl')
    datareader.init_word2vec('/home/tcastrof/Question/DiscoSumo/goeievraag/word2vec/word2vec.model')
    datareader.init_soft_cosine('/roaming/fkunnema/goeievraag/parsed/dict.model','/roaming/fkunnema/goeievraag/parsed/tfidf.model')
    # for server
    # datareader = datareader.Datareader(27017,'GoeieVraag','cqa','answers','qa_dict')
    # datareader.init_bm25('/var/lib/goeievraag/bm25.pkl')
    # datareader.init_word2vec('/var/lib/goeievraag/word2vec.model')
    # datareader.init_soft_cosine('/var/lib/goeievraag/dict.model','/var/lib/goeievraag/tfidf.model')
    print('DONE.')
    
    def get(self, request):

        return render(
            request,
            'template.html', {
                'retrieved':[],
                'placeholder':'Welke vraag heb je altijd al willen stellen?'
            }
        )

    def post(self, request):

        if 'Question' in request.POST:
            query = request.POST['Question']
            self.retrieved, self.duration = self.datareader.run(query)
        else:
            self.retrieved = []

        return render(
            request,
            'answer.html', {
                'retrieved':self.retrieved,
                'placeholder': query,
                'duur': self.duration
            }
        )
