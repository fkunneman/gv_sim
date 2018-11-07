
from django.shortcuts import render

import datareader

import sys



# Create your views here.




class GVHome(View):

    def get(self, request):

        self.datareader = datareader.Datareader()
        self.example_questions = self.datareader.return_example_questions()
        
        return render(
            request,
            'gv_sim/index.html', {
                'example_questions':self.example_questions
            }
        )

    def post(self, request):

        self.datareader = datareader.Datareader()
        self.example_questions = self.datareader.return_example_questions()
        
        return render(
            request,
            'gv_sim/index.html', {
                'example_questions':self.example_questions
            }
        )

    
class GVReturn(View):

    def get(self, request):

        return render(
            request,
            'gv_sim/search.html', {
                'query':'',
                'returned_questions':[],
                'returned_answers':[]
            }
        )


    def post(self, request):

        self.datareader = datareader.Datareader()
        self.returned_questions, self.returned_answers = datareader.rank_sims_bm25(request.POST['query'],similarity_threshold = 0.5) # similarity threshold value is used to classify question as similar / non-similar
        # self.returned_questions, self.returned_answers = datareader.rank_sims_soft_cosine(request.POST['query'],similarity_threshold = 0.5) # similarity threshold value is used to classify question as similar / non-similar

        return render(
            request,
            'gv_sim/search.html', {
                'query':request.POST['query'],
                'returned_questions':self.returned_questions,
                'returned_answers':self.returned_answers
            }
        )
    

        

        
