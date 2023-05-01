import codecs
from collections import defaultdict
import csv
import string
import numpy as np
from stop_words import get_stop_words

# Define a function to preprocess text data
def preprocess_text(text):
    # Replace newline characters with space
    text = text.replace('\n', ' ')
    # Remove punctuation from the text
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert text to lowercase
    text = text.lower()
    return text

# Load stop words for English
stop_words = get_stop_words('english')

# Create a dictionary to store admission IDs and their corresponding notes
admidic = defaultdict(list)
count = 0

# Read the NOTEEVENTS.csv file and store discharge summaries in admidic dictionary
with open('NOTEEVENTS.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in spamreader:
        if row[6] == 'Discharge summary':
            text = preprocess_text(row[-1])
            admidic[row[2]].append(text)
            count += 1

# Create a dictionary to count the frequency of each word in the notes
u = defaultdict(int)
for i in admidic:
    for text in admidic[i]:
        line = text.strip().split()
        for word in line:
            u[word] += 1

# Create a filtered dictionary with only words that occur more than 10 times and are not stop words
u2 = defaultdict(int)
for word in u:
    if not word.isdigit():
        if u[word] > 10:
            if word not in stop_words:
                u2[word] = u[word]

# Create a dictionary to store admission IDs and their corresponding ICD-9 codes
ad2c = defaultdict(list)
with codecs.open('DIAGNOSES_ICD.csv', 'r') as file1:
    line = file1.readline()
    line = file1.readline()
    while line:
        line = line.strip().split(',')
        if line[4][1:-1] != '':
            ad2c[line[2]].append('d_' + line[4][1:-1])
        line = file1.readline()

# Create a dictionary to count the frequency of each ICD-9 code
codeu = defaultdict(int)
for i in ad2c:
    for code in ad2c[i]:
        codeu[code] += 1

# Define a threshold for the minimum number of occurrences of each ICD-9 code
cthre = 0

# Write the preprocessed data to a new file called combined_dataset
with codecs.open('combined_dataset', 'w') as fileo:
    IDlist = np.load('IDlist.npy', encoding='bytes').astype(str)
    for i in IDlist:
        if ad2c[i]:
            fileo.write('start! ' + i + '\n')
            fileo.write('codes: ')
            tempc = []
            for code in ad2c[i]:
                if codeu[code] >= cthre:
                    if code[:5] not in tempc:
                        tempc.append(code[:5])
            for code in tempc:
                fileo.write(code + ' ')
            fileo.write('\n')
            fileo.write('notes:\n')
            for text in admidic[i]:
                thisline = text.strip().split()
                for word in thisline:
                    if u2[word] != 0:
                        fileo.write(word + ' ')
                fileo.write('\n')
            fileo.write('end!\n')
