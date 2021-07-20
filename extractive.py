from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import sumy
import heapq
import re
import networkx as nx
import numpy as np
from nltk.cluster.util import cosine_distance
from nltk.corpus import stopwords
import nltk

# pip install sumy

nltk.download('stopwords')
nltk.download('punkt')


def sentence_similarity_summarizer(text, numberof_top_sent=-1):
    def read_article(file_name):
        article = file_name.split(". ")
        sentences = []
        for sentence in article:
            # print(sentence)
            sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
        sentences.pop()
        return sentences

    def sentence_similarity(sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []

        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]

        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        # build the vector for the first sentence
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1

        # build the vector for the second sentence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1

        return 1 - cosine_distance(vector1, vector2)

    def build_similarity_matrix(sentences, stop_words):
        # Create an empty similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2:  # ignore if both are same sentences
                    continue
                similarity_matrix[idx1][idx2] = sentence_similarity(
                    sentences[idx1], sentences[idx2], stop_words)

        return similarity_matrix

    def generate_summary(file_name, top_n):
        stop_words = stopwords.words('english')
        summarize_text = []

        # Step 1 - Read text anc split it
        sentences = read_article(file_name)

        top_n = min(top_n, len(sentences))

        if top_n == -1:
            top_n = len(sentences) // 2

        # Step 2 - Generate Similary Martix across sentences
        sentence_similarity_martix = build_similarity_matrix(
            sentences, stop_words)

        # Step 3 - Rank sentences in similarity martix
        sentence_similarity_graph = nx.from_numpy_array(
            sentence_similarity_martix)
        scores = nx.pagerank(sentence_similarity_graph)

        # Step 4 - Sort the rank and pick top sentences
        ranked_sentence = sorted(
            ((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        #print("Indexes of top ranked_sentence order are ", ranked_sentence)

        for i in range(top_n):
            summarize_text.append(" ".join(ranked_sentence[i][1]))

        # Step 5 - Offcourse, output the summarize texr
        #print("Summarize Text: \n", ". ".join(summarize_text))
        return ". ".join(summarize_text)

    return generate_summary(text, numberof_top_sent)


def nltk_summarizer(text):
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', text)
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
    sentence_list = nltk.sent_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')
    word_frequencies = {}
    for word in nltk.word_tokenize(formatted_article_text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    maximum_frequncy = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequncy)
    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
    summary_sentences = heapq.nlargest(10,
                                       sentence_scores,
                                       key=sentence_scores.get)
    nltk_summary = ' '.join(summary_sentences)
    return nltk_summary


def lsa_summarizer(text):
    parser = PlaintextParser.from_string(text, Tokenizer('english'))
    lsa_summarizer = LsaSummarizer()
    lsa_summary = lsa_summarizer(parser.document, 10)
    lsa_summary_text = ''
    for sentence in lsa_summary:
        lsa_summary_text += str(sentence)
    return lsa_summary_text


def luhn_summarizer(text):
    parser = PlaintextParser.from_string(text, Tokenizer('english'))
    luhn_summarizer = LuhnSummarizer()
    luhn_summary = luhn_summarizer(parser.document, sentences_count=10)
    luhn_summary_text = ''
    for sentence in luhn_summary:

        luhn_summary_text += str(sentence)
    return luhn_summary_text


def kl_summarizer(text):
    parser = PlaintextParser.from_string(text, Tokenizer('english'))
    kl_summarizer = KLSummarizer()
    kl_summary = kl_summarizer(parser.document, sentences_count=10)
    kl_summary_text = ''
    for sentence in kl_summary:
        kl_summary_text += str(sentence)
    return kl_summary_text


def lexrank_summarizer(text):
    parser = PlaintextParser.from_string(text, Tokenizer('english'))
    lexrank_summarizer = LexRankSummarizer()
    lexrank_summary = lexrank_summarizer(parser.document, sentences_count=10)
    lexrank_summary_text = ''
    for sentence in lexrank_summary:
        lexrank_summary_text += str(sentence)
    return lexrank_summary_text


if __name__ == "__main__":
    text = input("Enter/Paste text: ")
    summaryType = input(
        "Type: \n 0: Sentence Similarity \n 1: NLTK Summarizer \n 2: LSA Summarizer \n 3: Luhn Summarizer \n 4: KL Summarizer \n 5: Lex-Rank Summarizer\n")

    summaryType = int(summaryType)

    if summaryType == 0:
        print(sentence_similarity_summarizer(text))
    elif summaryType == 1:
        print(nltk_summarizer(text))
    elif summaryType == 2:
        print(lsa_summarizer(text))
    elif summaryType == 3:
        print(luhn_summarizer(text))
    elif summaryType == 4:
        print(kl_summarizer(text))
    else:
        print(lexrank_summarizer(text))
