from nltk.translate.bleu_score import sentence_bleu

def bleau_score(references,candidate,ngram=1):
    references = [references.split()]
    candidate = candidate.split()
    assert ngram in [1,2,3,4]

    if ngram == 1:
        weight = (1, 0, 0, 0)
    elif ngram == 2:
        weight = (0.5, 0.5, 0, 0)
    elif ngram == 3:
        weight = (0.33,0.33,0.33,0)
    else:
        weight = (0.25,0.25,0.25,0.25)
    
    return sentence_bleu(references, candidate,weight)

