import fasttext 

model = fasttext.load_model('/workspace/datasets/fasttext/n_title_model.bin')
file = open('top_words.txt','r')
fout = open('synonyms.csv', 'w')
threshold = 0.75
for word in file.readlines():
    word = word.replace('\n', '')
    neighbors = [x[1] for x in list(filter(lambda x: x[0] >= threshold, model.get_nearest_neighbors(word)))]
    if len(neighbors) > 0:
        fout.write(word + ',' + ','.join(neighbors))
fout.close()