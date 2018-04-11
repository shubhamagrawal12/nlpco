import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


###nltk.download()

### tokenizing - word tokenizers, sentence tokenizers
### lexicon and corporas
### corpora - body of text. ex: medical journals, presidential speeches, English Language
### lexicon - words and their means

### investor speak - regular english speak
### investor speak 'bull' - someone who is positive about the market
### english speak 'bull' - scary animal you dont want running at you


#example_text = "Hello Mr. Smith, how are you doing today? The weather is great and Python is awesome. The sky is pinkish-blue. You should not eat cardboard."

#print(sent_tokenize(example_text))
#print(word_tokenize(example_text))   #list

#for i in word_tokenize(example_text):
#	print(i)

#example_sentence = "This is an example showing off stop words filtration."
#stop_words = set(stopwords.words("english"))

#print(stop_words)	

#words = word_tokenize(example_sentence)

##filtered_sentence = []

##for w in words:
##	if w not in stop_words:
##		filtered_sentence.append(w)

#filtered_sentence = [w for w in words if not w in stop_words]

#print(filtered_sentence)		

# I was taking a ride in the car.
# I was riding in the car.


ps = PorterStemmer()

example_words = ["python","pythoner","pythoning","pythoned","pythonly"]

#for w in example_words:
#	print(ps.stem(w))
	
new_text = "It is very important to be pythonly while you are pythoning with python. All pythoners have pythond poorly at least once."

words = word_tokenize(new_text)

for w in words:
	print(ps.stem(w))






