import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


#단어를 토큰화 함
def tokenize(sentence):
    
    return nltk.word_tokenize(sentence)


#Stemming으로 단어의 뿌리를 찾음
def stem(word):

    return stemmer.stem(word.lower())

"""
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
#존재하는 단어에 대해 bag을 만듬
def bag_of_words(tokenized_sentence, words):
    
    # 각 단어의 stem을 찾음
    sentence_words = [stem(word) for word in tokenized_sentence]
    # 모든 단어에 대한 bag리스트를 0으로 초기화
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
