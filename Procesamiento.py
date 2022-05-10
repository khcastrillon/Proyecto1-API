import nltk
from nltk.corpus import stopwords
import re
import pandas as pd
nltk.download('stopwords')

stop = set(stopwords.words('english'))
stop.remove('not')
sno = nltk.stem.SnowballStemmer('english')

# Funcion para limpiar texto
def cleanhtml(sentence): #function to clean the word of any html-tags
    cleanr = re.compile('<.*?>'.encode())
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext

# Funcion para limpiar texto
def cleanpunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]'.encode(),r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]'.encode(),r' ',cleaned)
    return  cleaned

def limpieza(data):
    datab = data
    str1 = ' '
    final_string = []
    s = ''
    for sent in datab.values:
        filtered_sentence = []
        # print(sent);
        sent = cleanhtml(sent)  # remove HTMl tags
        for w in sent.split():
            for cleaned_words in cleanpunc(w).split():
                if ((cleaned_words.isalpha()) & (len(cleaned_words) > 2)):
                    if (cleaned_words.lower() not in stop):
                        s = (sno.stem(cleaned_words.lower())).encode('utf8')
                        filtered_sentence.append(s)
                    else:
                        continue
                else:
                    continue
        str1 = b" ".join(filtered_sentence)  # final string of cleaned words
        # print("***********************************************************************")
        final_string.append(str1)
    return final_string