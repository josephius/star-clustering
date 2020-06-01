import codecs
import numpy as np
import nltk
from star_clustering import StarCluster

wordlist = nltk.corpus.words.words('en-basic')

print('Getting vectors...')
words = []
vectors = []
count = 0
with codecs.open('Data/wiki-news-300d-1M.vec', 'r', 'utf-8') as f:
    for i, line in enumerate(f):
        """
        if i > 0 and i <= 1000:
            data = line.strip().split()
            words.append(data[0])
            vectors.append(np.array([float(value) for value in data[1:]]))
        if i > 1000:
            break
        """
        word = line.strip().split()[0]
        if word in wordlist:
            words.append(word)
            vectors.append(np.array([float(value) for value in line.split()[1:]]))
            count += 1
            if count == len(wordlist):
                break
print(len(words))
vectors = np.array(vectors)

for name, algorithm in [('StarClustering', StarCluster())]:
    print(name)
    algorithm.fit(vectors, upper=True, limit_exp=-1, dis_type='angular')
    # algorithm.fit(vectors)
    labels = algorithm.labels_
    with codecs.open('basic_english_' + name + '.txt', 'w', 'utf-8') as f:
        for cluster in range(max(labels)+1):
            if cluster in labels:
                f.write(str(cluster) + ': ')
                for index in np.argwhere(labels == cluster):
                    f.write(words[int(index)] + ', ')
                f.write('\n')



