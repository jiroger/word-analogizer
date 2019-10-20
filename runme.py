from analogy import *

print("Please enter a maximum number of vocab words")
succeed = False
while(not succeed):
    try:
        max_vocab_words = int(input())
        succeed = True
    except:
        print("Please enter a valid maximum number of vocab words (> 0)")
succeed = False
print("Please enter a maximum number of context words")
while (not succeed):
    try:
        max_context_words = int(input())
        succeed = True
    except:
        print("Please enter a valid maximum number of context words (> 0)")
succeed = False
print("Hang on tight...this will take awhile!")
test = Analogy()
X_wiki = Analogy.generate_word_by_context(test.codes,
                              max_vocab_words=max_vocab_words,
                              max_context_words=max_context_words,
                              context_size=4,
                              weight_by_distance=True)
my_vectors = Analogy.reduce(Analogy.x_log(X_wiki), n_components=200)

# save in word2vec format (first line has vocab_size and dimension; other lines have word followed by embedding)
with codecs.open("my_vectors_200.txt", "w", "utf-8") as f:
    f.write(str(max_vocab_words) + " " + str(200) + "\n")
    for i in range(max_vocab_words):
        f.write(test.sorted_words[i] + " " + " ".join([str(x) for x in my_vectors[i,:]]) + "\n")

# load back in
word_vectors = KeyedVectors.load_word2vec_format("my_vectors_200.txt", binary=False)
while(True):
    print("Please enter the analogy you want to be solved in this format: x is to y as ____ is to b")
    print("What is x?")
    x = str(input())
    print("What is y?")
    y = str(input())
    print("What is b?")
    a = str(input())
    try:
        query = word_vectors.wv[x] - word_vectors.wv[y] + word_vectors.wv[a]
        print(word_vectors.wv.similar_by_vector(query))
    except:
        print("Hmm... looks like the analogizer can't find an answer. Try increasing the vocabulary size!")
