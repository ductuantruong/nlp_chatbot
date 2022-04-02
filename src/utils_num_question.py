from transformers import cached_path
import json

# url = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"

# personachat_file = cached_path(url)
# with open(personachat_file, "r", encoding="utf-8") as f:
#     dataset = json.loads(f.read())
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
def check_question(tokens):
  tagged = nltk.pos_tag(tokens)
  # if tagged[-1][0] == '?':
  #   return True
  for idx,tag in enumerate(tagged):
    if tag[0] == '?':
      return True
    if tag[1] in ['WRB','WP']:
      try:
        if tagged[idx+1][1] in ['VBZ','VBP','VBN','VBG','VBD','VB','MD']:
          return True
        if tagged[idx+1][0] in ['far','long','many','much','old','don’t','time']:
          return True
      except IndexError:
        continue
  return False

def question_calculate_val_dataset(valset):
  result = 0
  for conver in valset:
    utterances = conver['utterances']
    for index,candi_his in enumerate(utterances):
      if index == 0:
        continue
      previous_candi = utterances[index-1]['candidates']
      current_history = candi_his['history']
      # print(current_history[-1])
      if check_question(nltk.word_tokenize(current_history[-1])):
        result += 1
  return result

def question_calculate(lst):
  result = 0
  for senc in lst:
    if check_question(nltk.word_tokenize(senc)):
      print(senc)
      result += 1
  return result


