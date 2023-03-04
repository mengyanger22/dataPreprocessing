'''
# 从字典类型加载特征
measurements = [
    {"city" : "Dubai", "temperature" : 33.},
    {"city" : "London", "temperature" : 12.},
    {"city" : "San Francisco", "temperature" : 18.},
]

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
tmp = vec.fit_transform(measurements).toarray()
# print(tmp)
tmp = vec.get_feature_names_out()
# print(tmp)

pos_window = [
    {
        "word-2" : "the",
        "pos-2" : "DT",
        "word-1" : "cat",
        "pos-1" : "NN",
        "word+1" : "on",
        "pos+1" : "pp",
    },
]

vec = DictVectorizer()
pos_vectorized = vec.fit_transform(pos_window)
# print(pos_vectorized)
# print(pos_vectorized.toarray())
tmp = vec.get_feature_names_out()
print(tmp)
'''
'''
# 特征哈希（相当于一种降维技巧）
from sklearn.feature_extraction import FeatureHasher

def token_features(token, part_of_speech):
    if token.isdigit():
        yield "numeric"
    else:
        yield "token={}".format(token.lower())
        yield "token, pos={}, {}".format(token, part_of_speech)
    if token[0].isupper():
        yield "uppercase_initial"
    if token.isupper():
        yield "all_uppercase"
    yield "pos={}".format(part_of_speech)

# raw_X = (token_features(tok, pos_tagger(tok)) for tok in corpus)
# hasher = FeatureHasher(input_type="string")
# X = hasher.transform(raw_X)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
# print(vectorizer.__dict__)
corpus = [
    "This is the first document.",
    "This is the second second document.",
    "And the third one.",
    "Is this the first document?",
]

X = vectorizer.fit_transform(corpus)
# print(X)
# print(X.toarray())
analyze = vectorizer.build_analyzer()
tmp = analyze("This is two beautiful girl and handsome boy!")
# print(tmp)
tmp = vectorizer.get_feature_names_out()
# print(tmp)
# print(X.toarray())
tmp = vectorizer.vocabulary_.get("document")
# print(tmp)
tmp = vectorizer.transform(["Something completely new."]).toarray()
# print(tmp)

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r"\b\w+\b", min_df=1)
analyze = bigram_vectorizer.build_analyzer()
# print(analyze("I-am a fun guy!"))
X_2 = bigram_vectorizer.fit_transform(corpus).toarray()
# print(X_2)
feature_index = bigram_vectorizer.vocabulary_.get("is this")
# print(X_2[:, feature_index], feature_index)

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)
# print(transformer.__dict__)

counts = [
    [3, 0, 1],
    [2, 0, 0],
    [3, 0, 0],
    [4, 0, 0],
    [3, 2, 0],
    [3, 0, 2],
]

tfidf = transformer.fit_transform(counts)
# print(tfidf)
# print(tfidf.toarray())
# transformer = TfidfTransformer()
# print(transformer.fit_transform(counts).toarray())
# print(transformer.idf_)
# corpus = [
#     ["This is the first document."],
#     ["This is the second second document."],
#     ["And the third one."],
#     ["Is this the first document?"],
# ]
# from sklearn.feature_extraction.text import TfidfTransformer
# vectorizer = TfidfTransformer()
# tmp = vectorizer.fit_transform(corpus)
# # print(tmp)

import chardet

text1 = b"Sei mir gegr\xc3\xbc\xc3\x9ft mein Sauerkraut"
text2 = b"holdselig sind deine Ger\xfcche"
text3 = b"\xff\xfeA\x00u\x00f\x00 \x00F\x00l\x00\xfc\x00g\x00e\x00l\x00n\x00 \x00d\x00e\x00s\x00 \x00G\x00e\x00s\x00a\x00n\x00g\x00e\x00s\x00,\x00 \x00H\x00e\x00r\x00z\x00l\x00i\x00e\x00b\x00c\x00h\x00e\x00n\x00,\x00 \x00t\x00r\x00a\x00g\x00 \x00i\x00c\x00h\x00 \x00d\x00i\x00c\x00h\x00 \x00f\x00o\x00r\x00t\x00"
decoded = [x.decode(chardet.detect(x)["encoding"]) for x in (text1, text2, text3)]
v = CountVectorizer().fit(decoded).vocabulary_
# for term in v:
#     print(v)

# import numpy as np
# from sklearn.decomposition import NMF

# X = np.random.random((3, 4))
# print(X)
# model = NMF()
# W = model.fit_transform(X)
# H = model.components_
# print(W)
# print(H)

# X = np.random.random((4, 3))
# model = NMF(5)
# W = model.fit_transform(X)
# H = model.components_
# print(W)
# print(W.shape)
# print(H.shape)


# ngram_vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(2, 2))
# counts = ngram_vectorizer.fit_transform(["words", "wprds"])
# tmp = ngram_vectorizer.get_feature_names_out()
# print(tmp)
# print(counts.toarray())

# ngram_vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(5, 5))
# counts = ngram_vectorizer.fit_transform(["jumpy fox"])
# tmp = ngram_vectorizer.get_feature_names_out()
# # print(tmp)
# # print(counts.toarray())
# ngram_vectorizer = CountVectorizer(analyzer="char", ngram_range=(5, 5))
# counts = ngram_vectorizer.fit_transform(["jumpy fox"])
# features = ngram_vectorizer.get_feature_names_out()
# print(features)
# print(counts.toarray())
# counts = ngram_vectorizer.transform(["jxmpy"])
# print(counts.toarray())

# from sklearn.feature_extraction.text import HashingVectorizer

# hv = HashingVectorizer(n_features=10)
# tmp = hv.transform(corpus)
# print(tmp.toarray())
'''
'''
import itertools
from pathlib import Path
from hashlib import sha256
import re
import tarfile
import time
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from html.parser import HTMLParser
from urllib.request import urlretrieve

from sklearn.datasets import get_data_home
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.naive_bayes import MultinomialNB


def _not_in_sphinx():
    return "__file__" in globals()

class ReutersParser(HTMLParser):
    def __init__(self, encoding="latin-1"):
        HTMLParser.__init__(self)
        self._reset()
        self.encoding = encoding
    
    def handle_starttag(self, tag, attrs):
        method = "start_" + tag
        getattr(self, method, lambda x:None)(attrs)

    def handle_endtag(self, tag):
        method = "end_" + tag
        getattr(self, method, lambda:None)()
    
    def _reset(self):
        self.in_title = 0
        self.in_body = 0
        self.in_topics = 0
        self.in_topics_d = 0
        self.title = ""
        self.body = ""
        self.topics = []
        self.topics_d = ""
    
    def parse(self, fd):
        self.docs = []
        for chunk in fd:
            self.feed(chunk.decode(self.encoding))
            for doc in self.docs:
                yield doc
            self.docs = []
        self.close()

    def handle_data(self, data):
        if self.in_body:
            self.body += data
        elif self.in_title:
            self.title += data
        elif self.in_topics_d:
            self.topics_d += data
    
    def start_reuters(self, attributes):
        pass

    def end_reuters(self):
        self.body = re.sub(r"\s+", r" ", self.body)
        self.docs.append({
            "title" : self.title,
            "body" : self.body,
            "topics" : self.topics,
        })
        self._reset()

    def start_title(self, attributes):
        self.in_title += 1
    
    def end_title(self):
        self.in_title = 0
    
    def start_body(self, attributes):
        self.in_body += 1
    
    def end_body(self):
        self.in_body = 0
    
    def start_topics(self, attributes):
        self.in_topics += 1
    
    def end_topics(self):
        self.in_topics = 0
    
    def start_d(self, attributes):
        self.in_topics_d = 1
    
    def end_d(self):
        self.in_topics = 0
        self.topics.append(self.topics_d)
        self.topics_d = ""


def stream_reuters_documents(data_path=None):
    DOWNLOAD_URL = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/reuters21578.tar.gz"
    )
    ARCHIVE_SHA256 = "3bae43c9b14e387f76a61b6d82bf98a4fb5d3ef99ef7e7075ff2ccbcf59f9d30"
    ARCHIVE_FILENAME = "reuters21578.tar.gz"

    if data_path is None:
        data_path = Path(get_data_home()) / "reuters"
    else:
        data_path = Path(data_path)
    
    if not data_path.exists():
        print("downloading dataset (one and for all) into %s" % data_path)
        data_path.mkdir(parents=True, exist_ok=True)

        def progress(blocknum, bs, size):
            total_sz_mb = "%.2f MB" % (size / 1e6)
            current_sz_mb = "%.2f MB" % ((blocknum * bs) / 1e6)
            if _not_in_sphinx():
                sys.stdout.write("\rdownloaded %s / %s" % (current_sz_mb, total_sz_mb))
        archive_path = data_path / ARCHIVE_FILENAME

        urlretrieve(DOWNLOAD_URL, filename=archive_path, reporthook=progress)
        if _not_in_sphinx():
            sys.stdout.write("\r")

        # Check that the archive was not tampered:
        assert sha256(archive_path.read_bytes()).hexdigest() == ARCHIVE_SHA256

        print("untarring Reuters dataset...")
        tarfile.open(archive_path, "r:gz").extractall(data_path)
        print("done.")

    parser = ReutersParser()
    for filename in data_path.glob("*.sgm"):
        for doc in parser.parse(open(filename, "rb")):
            yield doc



def main():
    vectorizer = HashingVectorizer(
        decode_error= "ignore", n_features=2**18, alternate_sign=False
    )
    data_stream = stream_reuters_documents()
    all_classes = np.array([0, 1])
    positive_class = "acq"

    partial_fit_classifiers = {
        "SGD" : SGDClassifier(max_iter=5),
        "Perceptron" : Perceptron(),
        "NB Multinomial" : MultinomialNB(alpha=0.01),
        "Passive-Aggressive" : PassiveAggressiveClassifier(),
    }

    def get_minibatch(doc_iter, size, pos_class=positive_class):
        data = [
            ("{title}\n\n{body}".format(**doc), pos_class in doc["topics"])
            for doc in itertools.islice(doc_iter, size)
            if doc["topics"]
        ]
        if not len(data):
            return np.asarray([], dtype=int), np.asarray([], dtype=int)
        X_test, y = zip(*data)
        return X_test, np.asarray(y, dtype=int)
    
    def iter_minibatches(doc_iter, minibatch_size):
        X_test, y = get_minibatch(doc_iter, minibatch_size)
        while len(X_test):
            yield X_test, y
            X_test, y = get_minibatch(doc_iter, minibatch_size)

    test_stats = {"n_test":0, "n_test_pos":0}
    n_test_documents = 1000
    tick = time.time()
    X_test_text, y_test = get_minibatch(data_stream, 1000)
    parsing_time = time.time() - tick
    tick = time.time()
    X_test = vectorizer.transform(X_test_text)
    vectorizing_time = time.time() - tick
    test_stats["n_test"] += len(y_test)
    test_stats["n_test_pos"] += sum(y_test)
    print("Test set is %d documents (%d positive)" % (len(y_test), sum(y_test)))

    def progress(cls_name, stats):
        duration =  time.time() - stats["t0"]
        s = "%20s classifier : \t" % cls_name
        s += "%(n_train)6d train docs (%(n_train_pos)6d positive) " % stats
        s += "%(n_test)6d test docs (%(n_test_pos)6d positive) " % test_stats
        s += "accuracy: %(accuracy).3f " % stats
        s += "in %.2fs (%5d docs/s)" % (duration, stats["n_train"] / duration)
        return s
    
    cls_stats = {}
    for cls_name in partial_fit_classifiers:
        stats = {
            "n_train" : 0,
            "n_train_pos" : 0,
            "accuracy" : 0.0,
            "accuracy_history" : [(0, 0)],
            "t0" : time.time(),
            "runtime_history" : [(0, 0)],
            "total_fit_time" : 0.0,
        }
        cls_stats[cls_name] = stats
    
    get_minibatch(data_stream, n_test_documents)
    minibatch_size = 1000
    minibatch_iterators = iter_minibatches(data_stream, minibatch_size)
    total_vect_time = 0.0
    for i, (X_train_text, y_train) in enumerate(minibatch_iterators):
        tick = time.time()
        X_train = vectorizer.transform(X_train_text)
        total_vect_time += time.time() - tick
        for cls_name, cls in partial_fit_classifiers.items():
            tick = time.time()
            cls.partial_fit(X_train, y_train, classes=all_classes)
            cls_stats[cls_name]["total_fit_time"] += time.time() - tick
            cls_stats[cls_name]["n_train"] += X_train.shape[0]
            cls_stats[cls_name]["n_train_pos"] += sum(y_train)
            tick = time.time()
            cls_stats[cls_name]["accuracy"] = cls.score(X_test, y_test)
            cls_stats[cls_name]["prediction_time"] = time.time() - tick
            acc_history = (cls_stats[cls_name]["accuracy"], cls_stats[cls_name]["n_train"])
            cls_stats[cls_name]["accuracy_history"].append(acc_history)
            run_history = (
                cls_stats[cls_name]["accuracy"],
                total_vect_time + cls_stats[cls_name]["total_fit_time"]
            )
            cls_stats[cls_name]["runtime_history"].append(run_history)
            if i % 3 == 0:
                print(progress(cls_name, cls_stats[cls_name]))
        if i % 3 == 0:
            print("\n")


if __name__ == "__main__":
    main()

'''
# from sklearn.feature_extraction.text import CountVectorizer

# def my_tokenizer(s):
#     return s.split()

# vectorizer = CountVectorizer(tokenizer=my_tokenizer)
# tmp = vectorizer.build_analyzer()(u"some... puncuation!")
# print(tmp)

# from sklearn.base import BaseEstimator

# print(dir(BaseEstimator.__init__))
from sklearn.base import BaseEstimator
import inspect

# init = getattr(BaseEstimator.__init__, "deprecated_original", "none")
# print(init)

class A:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def test(self, c, d):
        print(c, d)

a = A("a", "b")
res = inspect.signature(a.test)
print(res)