from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn import preprocessing
from nltk.corpus import stopwords
import string
from keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer

question_model = T5ForConditionalGeneration.from_pretrained(
    'ramsrigouthamg/t5_squad_v1')
question_tokenizer = T5Tokenizer.from_pretrained('t5-base')

tokenizer = AutoTokenizer.from_pretrained(
    "/Users/saiganeshthamaraikannan/Desktop/Project/gpt3")
model = AutoModelForCausalLM.from_pretrained(
    "/Users/saiganeshthamaraikannan/Desktop/Project/gpt3")


def cluster(learning_resources):
    tokenizer = Tokenizer()
    stop_words = stopwords.words('english')
    no_of_lr = len(learning_resources)
    punctuation_removed_lr = []
# remove punctuation
    for i in learning_resources:
        punctuation_removed_lr.append(
            "".join([char for char in i if char not in string.punctuation]))


# tokenize words

    def tokenize(word):
        tokenizer.fit_on_texts([word])
        word_index = tokenizer.word_index
        sequences = tokenizer.texts_to_sequences([word])
        words = tokenizer.word_index.keys()
        return words

    words_tokens = []
# calling tokenizer to remove stopwords and word into lowercase
    for i in punctuation_removed_lr:
        words = [t.lower() for t in tokenize(i) if t.lower() not in stop_words]
        words_tokens.append(' '.join(words))

    vectorizer = TfidfVectorizer()
    tfidf_vectors = vectorizer.fit_transform(words_tokens)

# k-means clustering

    inertias = []

# Try k values from 1 to 10
    for k in range(1, no_of_lr+1):
        # Fit K-Means with k clusters
        kmeans = KMeans(n_clusters=k, random_state=0).fit(tfidf_vectors)
    # Add inertia to list
        inertias.append(kmeans.inertia_)
# Choose optimal k value based on elbow curve
    diff = np.diff(inertias)
    diff2 = np.diff(diff)
    elbow_index = np.where(diff2 > 0)[0][0] + 2
    optimal_k = elbow_index

# Fit K-Means with optimal k value
    kmeans = KMeans(n_clusters=optimal_k, random_state=0, n_init=20).fit(
        preprocessing.normalize(tfidf_vectors))
    clusters = {}
    for i, label in enumerate(kmeans.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(learning_resources[i])
    return clusters

# root-keyphrase extraction


def root_key_phrase(keywords, pr_lr):
    cosine_values = dict()
    vectorizer = TfidfVectorizer()
    for kw in keywords:
        for lr in pr_lr:
            kv = vectorizer.fit_transform([kw, lr])
            similarity = cosine_similarity(kv[0], kv[1])[0][0]
            if kw not in cosine_values:
                cosine_values[kw] = similarity
            else:
                cosine_values[kw] += similarity
        cosine_values[kw] = cosine_values[kw]/len(pr_lr)

    return max(cosine_values, key=cosine_values.get)


def extract_keywords(learningresource, model):
    keywords = model.extract_keywords(learningresource, keyphrase_ngram_range=(
        1, 2), stop_words='english', top_n=5, vectorizer=KeyphraseCountVectorizer())
    return keywords


def get_keys(segment):
    model = KeyBERT()
    key_words = []
    each_lr_keywords = dict()
    for i, lr in enumerate(segment):
        each_lr_keywords[i] = [k[0] for k in extract_keywords(lr, model)]
        key_words.extend(each_lr_keywords[i])
    root_key_word = root_key_phrase(key_words, segment)
    return [root_key_word, key_words]


def definition(word):
    input_text = word
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids=input_ids,
                            max_length=100, temperature=0.7)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_output = generated_text.split('\n')[:2]
    return ' '.join(generated_output)


def generate_overview(root_key, root_key_def, definitions):
    overview = f'This explains {root_key}.'
    overview += root_key_def
    overview += 'this topic includes '
    for i in definitions.keys():
        overview += ','+i
    overview += '.'
    return overview


def get_question(sentence, answer):
    text = "context: {} </s>".format(sentence, answer)
    print(text)
    max_len = 256
    encoding = question_tokenizer.encode_plus(
        text, max_length=max_len, pad_to_max_length=True, return_tensors='pt')
    input_ids, attention_mask = encoding['input_ids'], encoding['attention_mask']
    outs = question_model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                   early_stopping=True, num_beams=5,
                                   num_return_sequences=1, no_repeat_ngram_size=2,
                                   max_length=200
                                   )
    doc = [question_tokenizer.decode(ids) for ids in outs]
    Question = doc[0].replace("question :", "")
    Question = Question.strip().replace('<\s>', '').replace('<pad>', '')
    return Question


def generate_questions(definitions):
    answers = definitions.keys()
    questions = set()
    for ans in answers:
        sentence = definitions[ans].split('.')[0]
        if sentence.strip() != '':
            questions.add(get_question(sentence, ans).lower())
    return questions


def preprocess(lr):
    learning = []
    for i in lr:
        t = i.strip()
        if t != '':
            learning.append(t)
    return learning


def overview(lr):
    learning_resource = []
    learning_resource = preprocess(lr)
    clusters = dict()
    if len(learning_resource) > 2:
        clusters = cluster(learning_resource)
    else:
        out = 1
        for i in learning_resource:
            clusters[out] = []
            clusters[out].append(i)
            out += 1
    overviews = []
    questions = []
    for i, clust in enumerate(clusters.values()):
        over = f'overview-{i+1}: \n'
        root_key, keywords = get_keys(clust)
        root_key_def = definition(root_key)
        definitions = dict()
        for key in keywords:
            definitions[key] = definition(key)
        overviews.append(over+generate_overview(
            root_key, root_key_def, definitions))
        questions.extend(generate_questions(definitions))

    return [overviews, questions]


learning_resources = ["What is natural language processing? Natural language processing (NLP) refers to the branch of computer science—and more specifically, the branch of artificial intelligence or AI—concerned with giving computers the ability to understand text and spoken words in much the same way human beings can",
                      "Machine learning is an important component of the growing field of data science. Through the use of statistical methods, algorithms are trained to make classifications or predictions, and to uncover key insights in data mining projects. These insights subsequently drive decision making within applications and businesses, ideally impacting key growth metrics. As big data continues to expand and grow, the market demand for data scientists will increase. They will be required to help identify the most relevant business questions and the data to answer them.",
                      "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to intelligence of humans and other animals. Example tasks in which this is done include speech recognition, computer vision, translation between (natural) languages, as well as other mappings of inputs."]
