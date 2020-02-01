import os
import re
import math

from nltk import PorterStemmer
from collections import Counter
from statistics import mean

# Declare directory paths and variables
directory_path = '/Users/asmitasingla/Desktop/MSCSS/Quarter-2/Information-Retrieval/HW/transcripts'
stopwords_file_path = '/Users/asmitasingla/Desktop/MSCSS/Quarter-2/Information-Retrieval/HW/inputfiles/stopwords.txt'
wordsBeforeProcessing = []
noStopWords = []
stemmedWordsList = []
combinedText = []
totalWordsPerDocument = []
noOfDocWithTerm = []
dictOfWords = []
listOfWordsOccurringOncePerDoc = []
listOfWordsOccurringOnce = []

# Create the list of stop words from the given stopwords text file
stopwords_file = open(stopwords_file_path, "r")
stopword_list = stopwords_file.read().splitlines()
stopwords_file.close()

# Read all the text files and create a list
for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        complete_file_path = os.path.join(directory_path, filename)
        file1 = open(complete_file_path)
        text = file1.read()
        combinedText.append(text)

# Create a stemmed dictionary of words mapped per document to calculate df
stemWord = PorterStemmer()
stemmedDict = {}
counter = 1
for line in combinedText:
    line = re.sub('[^A-Za-z0-9\s]+', ' ', line)
    stemmedDict[counter] = []
    for word in line.lower().split():
        stemmedDict[counter].append(stemWord.stem(word))
    counter += 1

# Join all the text files into one string
joinedString = ' '.join(map(str, combinedText))

# Remove special characters from the string
strWithNoSpecial = re.sub('[^A-Za-z0-9\s]+', ' ', joinedString)

# Tokenize the words in the string
tokenizedWords = strWithNoSpecial.lower().split()

for w in tokenizedWords:
    # Append all the words to a list to get a word token count before processing the data
    wordsBeforeProcessing.append(w)

    # Remove stop words and save in a list
    if w not in stopword_list:
        noStopWords.append(w)

# Stem the words
for word in noStopWords:
    stem_word = PorterStemmer().stem(word)
    stemmedWordsList.append(stem_word)

for i in stemmedDict:
    totalWordsPerDocument.append(len(stemmedDict[i]))

# Dictionary of word tokens before removing duplicates
dictOfWords = [(word, stemmedWordsList.count(word)) for word in stemmedWordsList]

# Calculate no of words occurring once in the database
for wordFreq in dictOfWords:
    if wordFreq[1] == 1:
        listOfWordsOccurringOncePerDoc.append(wordFreq)
listOfWordsOccurringOnce = [ls[0] for ls in listOfWordsOccurringOncePerDoc]

# Get 30 most frequent words with the TF
wordCounter = Counter(stemmedWordsList)
mostOccurringTf = wordCounter.most_common(30)
listOfTop30 = [(term[0]) for term in mostOccurringTf]
print('30 most frequent stemmed words:', listOfTop30)
print('Term Frequency of top 30 words', mostOccurringTf)

# Calculate the Term Frequency Weight
weightedTf = [(term[0], 1 + math.log10(term[1])) for term in mostOccurringTf]
print('Term Frequency weight of top 30 words', weightedTf)

# Calculate the normalized Term Frequency
normalisedTfList = {}
for term in mostOccurringTf:
    normalisedTfList[term[0]] = term[1] / len(mostOccurringTf)
print('Normalised Term Frequency of top 30 words', normalisedTfList)

# Calculate Document frequency of top 30 words
dfCountDict = {}
for word in listOfTop30:
    dfCountDict[word] = 0
    for voc in stemmedDict:
        if word in stemmedDict[voc]:
            dfCountDict[word] += 1
print('Document frequency of top 30 words', dfCountDict)

# Calculate Inverse Document frequency of top 30 words
for term in dfCountDict:
    dfCountDict[term] = math.log10(len(combinedText) / dfCountDict[term])
print('Inverse document Frequency of top 30 words', dfCountDict)

# Calculate Term Frequency * Inverse Document frequency of top 30 words
tfIdfValueDict = {}
for term in normalisedTfList:
    tfIdfValueDict[term] = normalisedTfList[term] * dfCountDict[term]
print('TF * IDF of top 30 words', tfIdfValueDict)

# Calculate probability of term
termProbabilityList = [(term[0], term[1] / len(stemmedWordsList)) for term in mostOccurringTf]
print('Term probability of top 30 words:', termProbabilityList)

# Printing the results
print('The number of word tokens before text processing:', len(wordsBeforeProcessing))
print('The number of word tokens after text processing:', len(stemmedWordsList))
print('The number of unique word tokens:', len(set(stemmedWordsList)))
print('The number of words that occur once in database:', len(set(listOfWordsOccurringOnce)))
print('The average number of words per document:', round(mean(totalWordsPerDocument)))

