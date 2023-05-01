import codecs
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# initialize empty dictionaries for the Wikipedia and notes vocabularies
wikivocab = {}

# open and read from the "wikipedia_knowledge" file
file1 = codecs.open("wikipedia_knowledge", "r", "utf-8")
line = file1.readline()
while line:
 # if the line does not start with "XXX", it contains text data
    if line[0:3] != "XXX":
    # preprocess the line by removing newline characters and splitting it into words
        line = line.strip("\n")
        line = line.split()
     # add each word in the line to the wikivocab dictionary in lowercase form

        for i in line:
            wikivocab[i.lower()] = 1
    line = file1.readline()

notesvocab = {}

# open and read from the "combined_dataset" file
filec = codecs.open("combined_dataset", "r", "utf-8")
line = filec.readline()
while line:
    line = line.strip("\n")
    line = line.split()
    if line[0] == "codes:":
        line = filec.readline()
        line = line.strip("\n")
        line = line.split()
        if line[0] == "notes:":
            line = filec.readline()
            while line != "end!\n":
                line = line.strip("\n")
                line = line.split()
                for word in line:
                    notesvocab[word] = 1
                line = filec.readline()
    line = filec.readline()

# get the intersection of the notes and Wikipedia vocabularies
a1 = set(notesvocab)
a2 = set(wikivocab)
a3 = a1.intersection(a2)

# initialize empty lists for the Wikipedia and notes documents
wikidocuments = []

# open and read from the "wikipedia_knowledge" file
file2 = codecs.open("wikipedia_knowledge", "r", "utf-8")
line = file2.readline()
while line:
    if line[0:4] == "XXXd":
        tempf = []
        line = file2.readline()
        while line[0:4] != "XXXe":
            line = line.strip("\n")
            line = line.split()
            for i in line:
                if i.lower() in a3:
                    tempf.append(i.lower())
            line = file2.readline()
        wikidocuments.append(tempf)
    line = file2.readline()

# Read in the combined dataset file
notesdocuments = []
file3 = codecs.open("combined_dataset", "r", "utf-8")
line = file3.readline()

# Extract notes section from file
while line:
    line = line.strip("\n")
    line = line.split()
    if line[0] == "codes:":
        line = file3.readline()
        line = line.strip("\n")
        line = line.split()
        if line[0] == "notes:":
            tempf = []
            line = file3.readline()
            while line != "end!\n":
                line = line.strip("\n")
                line = line.split()
                for word in line:
                    if word in a3:
                        tempf.append(word)
                line = file3.readline()
            notesdocuments.append(tempf)
    line = file3.readline()

# Create vocabulary from notes data
notesvocab = {}
for i in notesdocuments:
    for j in i:
        if j.lower() not in notesvocab:
            notesvocab[j.lower()] = len(notesvocab)
            
# Convert notes data into binary vector
notedata = []
for i in notesdocuments:
    temp = ""
    for j in i:
        temp = temp + j + " "
    notedata.append(temp)
    
vect = CountVectorizer(min_df=1, vocabulary=notesvocab, binary=True)
binaryn = vect.fit_transform(notedata)
binaryn = binaryn.A
binaryn = np.array(binaryn, dtype=float)

# Convert wikipedia data into binary vector
wikidata = []
for i in wikidocuments:
    temp = ""
    for j in i:
        temp = temp + j + " "
    wikidata.append(temp)

vect2 = CountVectorizer(min_df=1,vocabulary=notesvocab,binary=True)
binaryk = vect2.fit_transform(wikidata)
binaryk=binaryk.A
binaryk=np.array(binaryk,dtype=float)

# Save binary vectors as numpy arrays
np.save('notevec',binaryn)
np.save('wikivec',binaryk)



