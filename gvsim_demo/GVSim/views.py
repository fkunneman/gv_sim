
from django.shortcuts import render
from django.views.generic import View

from . import datareader

import sys



# Create your views here.




class GVHome(View):

    def get(self, request):

        self.datareader = datareader.Datareader(27020,'GoeieVraag','cqa','examples','answers','qa_dict')

        return render(
            request,
            'index.html', {
                'stage':'start',
                'retrieved':[],
                'example_questions':self.datareader.examples,
                'placeholder':'Welke vraag heb je altijd al willen stellen?'
            }
        )

    def post(self, request):

        self.datareader = datareader.Datareader(27020,'GoeieVraag','cqa','examples','answers','qa_dict')

        self.bm25_path = '/roaming/fkunnema/goeievraag/exp_similarity_new/bm25.pkl'
        
        if 'Question' in request.POST:
            query = request.POST['Question']
            self.retrieved = self.datareader.rank_sims_bm25(self.bm25_path, query, 0.2)
        else:
            self.retrieved = []

        return render(
            request,
            'index.html', {
                'stage':'search',
                'retrieved':self.retrieved,
                'example_questions':self.datareader.examples,
                'placeholder': query
            }
        )
