##########  LDA  ##########

import logging
import pyLDAvis.gensim
import numpy as np
from gensim import corpora, similarities
from gensim.models import ldamodel
from time import time 
from math import log

# читаем файл с токенизированными и лемматизированными текстами объявлений, разбиваем тексты на слова, считаем стреднюю длину текста
texts = []
with open('textsMC.txt', 'r', encoding='utf-8') as f:
    for line in f:
        if not line.startswith('#'):
            texts.append([w for w in line.strip('\n').split()])
print(len(texts), np.mean([len(text) for text in texts]))

# векторизуем тексты
dictionary = corpora.Dictionary(texts)
print('Original: {}'.format(dictionary))
dictionary.filter_extremes(no_below = 5, no_above = 0.5, keep_n=None)
print('Filtered: {}'.format(dictionary))
corpus = [dictionary.doc2bow(text) for text in texts]

# строим модель
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO
%time lda = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=100, chunksize=100, update_every=1, passes=2)
# если увеличить passes и num_topics, модель будет дольше обучаться, но и перплексия будет ниже (т.е. лучше)

# считаем перплексию
def perplexity(model, corpus):
    corpus_length = 0
    log_likelihood = 0
    topic_profiles = model.state.get_lambda() / np.sum(model.state.get_lambda(), axis=1)[:, np.newaxis]
    for document in corpus:
        gamma, _ = model.inference([document])
        document_profile = gamma / np.sum(gamma)
        for term_id, term_count in document:
            corpus_length += term_count
            term_probability = np.dot(document_profile, topic_profiles[:, term_id])
            log_likelihood += term_count * log(term_probability)
    perplexity = np.exp(-log_likelihood / corpus_length)
    return perplexity
print('Перплексия: ', perplexity(lda, corpus))

# топ-10 слов для каждого топика можно просмотреть, раскомментировав следующие строки:
# top_words = [[word for _, word in lda.show_topic(topicNumber, topn=10)] for topicNumber in range(lda.num_topics)]
# for i in range(len(top_words)):
#     print(i+1, ' '.join(top_words[i]), '\n')

# визуализируем топики (карта расстояния между топиками) и топ-30 слов
vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.display(vis)








##########  K-means  ########## 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

docs = []
true_k = 6 # количество исходных категорий

# читаем файл с токенизированными и лемматизированными текстами объявлений
with open('textsMC.txt', 'r', encoding='utf-8') as f:
    for line in f:
        if not line.startswith('#'):
            docs.append(line.strip('\n'))

# строим матрицу tf-idf
vectorizer = TfidfVectorizer(ngram_range=(1,1), max_df=0.8, min_df=1)
matrix = vectorizer.fit_transform(docs)

# кластеризуем
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(matrix)

# для каждого кластера выводим топ-10 близких к центру кластера слов
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(1, true_k+1):
    print("Cluster %d:" % i)
    top = []
    for ind in order_centroids[i-1, :10]:
        top.append(terms[ind])
    print(' '.join(top))








##########  Visualize k-means + categories  ########## 

import mpld3
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

# если в ipytho notebook, раскоментируйте следующую строчку
# %matplotlib inline
matplotlib.style.use('ggplot')
plt.rcParams['figure.figsize'] = (17, 10)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12


categories = []
docs = []
true_k = 6

# читаем файл с токенизированными и лемматизированными текстами объявлений
with open('textsMC+category.txt', 'r', encoding='utf-8') as f:
    for line in f: #[next(f) for i in range(7775)]:
        if not line.startswith('#'):
            if line.startswith('<'):
                cats.append(line.strip('\n').split()[1])
            else:
                docs.append(line.strip('\n'))

# если имена категорий написаны кириллицей, заменяем на эквивалентные имена латиницей
for rus, eng in zip(['выставки', 'кинопоказы', 'концерты', 'лекции', 'спектакли', 'фестивали'], 
                    ['exhibitions', 'films', 'concerts', 'lectures', 'shows', 'festivals']):
    categories = [c.replace(rus, eng) for c in categories]

# строим матрицу tf-idf
vectorizer = TfidfVectorizer(ngram_range=(1,1), max_df=0.8, min_df=1)
matrix = vectorizer.fit_transform(docs)
terms = vectorizer.get_feature_names()
dist = 1 - cosine_similarity(matrix)

# кластеризуем
model = KMeans(n_clusters=true_k)
%time model.fit(matrix)
clusters = model.labels_.tolist()

# записываем тексты, категории и получившиеся кластеры в data frame
info = {'category': categories, 'texts': docs, 'cluster': clusters}
infoDF = pd.DataFrame(info, index = [clusters] , columns = ['category', 'cluster'])

# для каждого кластера выводим на экран топ-10 слов и сколько текстов какой категории попало в данный кластер
order_centroids = model.cluster_centers_.argsort()[:, ::-1] 
for i in range(true_k):
    print("Cluster %d" % i)
    top = []
    for ind in order_centroids[i, :10]:
        top.append(terms[ind])
    print('Top words: ' + '\n\t' + ' '.join(top))
    topCat = []
    count = []
    for category in infoDF.ix[i]['category'].values.tolist():
        topCat.append(category)
    print('Categories:')
    for cat in set(topCat):
        count.append([topCat.count(cat), cat])
    for item in [i[1]+': '+str(i[0]) for i in sorted(count, reverse=True)]:
        print('\t'+item)

# используем многомерное шкалирование для последующей визуализации
MDS()
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
pos = mds.fit_transform(dist) 
xs, ys = pos[:, 0], pos[:, 1]
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#4878CF', 3: '#F0E442', 4: '#343786', 5: '#722717'}
cluster_names = {0: 'plays', 
                 1: 'exhibitions', 
                 2: 'concerts', 
                 3: 'lectures', 
                 4: 'festivals',
                 5: 'films'}

# записывем в data frame и группироуем по кластерам
onemoredf = pd.DataFrame(dict(x=xs, y=ys, label=clusters, category=categories)) 
groups = onemoredf.groupby('label')

# javascript и css для экспорта в html-формат и для форматирования
class TopToolbar(mpld3.plugins.PluginBase):
    """Plugin for moving toolbar to top of figure"""

    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };

    TopToolbar.prototype.draw = function(){
      // the toolbar svg doesn't exist
      // yet, so first draw it
      this.fig.toolbar.draw();

      // then change the y position to be
      // at the top of the figure
      this.fig.toolbar.toolbar.attr("x", 150);
      this.fig.toolbar.toolbar.attr("y", 400);

      // then remove the draw function,
      // so that it is not called again
      this.fig.toolbar.draw = function() {}
    }
    """
    def __init__(self):
        self.dict_ = {"type": "toptoolbar"}

css = """
text.mpld3-text, div.mpld3-tooltip {
  font-family:Arial, Helvetica, sans-serif;
}
g.mpld3-xaxis, g.mpld3-yaxis {
display: none; }
svg.mpld3-figure {
margin-left: -200px;}
"""

# визуализируем
fig, ax = plt.subplots(figsize=(17,7)) 
ax.margins(0.03) 
for name, group in groups:
    points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=10, 
                     label=cluster_names[name], mec='none', 
                     color=cluster_colors[name])
    ax.set_aspect('auto')
    labels = [i for i in group.category]  
    tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels, voffset=10, hoffset=10, css=css)
    mpld3.plugins.connect(fig, tooltip, TopToolbar())    
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
ax.legend(numpoints=1)
mpld3.display()

# если требуется html код визуализации, раскоментируйте следующую строку
# html = mpld3.fig_to_html(fig)









######### t-SNE  ##########

from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

docs = []

# читаем файл с токенизированными и лемматизированными текстами объявлений
with open('textsMC.txt', 'r', encoding='utf-8') as f:
    for line in f:
        if not line.startswith('#'):
            docs.append(line.strip('\n'))

# строим матрицу tf-idf
vectorizer = TfidfVectorizer(ngram_range=(1,1), max_df=0.8, min_df=1)
matrix = vectorizer.fit_transform(docs)

# обучаем модель
X_reduced = TruncatedSVD(n_components=50, random_state=0).fit_transform(matrix)
X_embedded = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(X_reduced)

for eng, num in zip(['exhibitions', 'films', 'concerts', 'lectures', 'plays', 'festivals'], ['1','2','3','4','5','6']):
    categories = [c.replace(eng, num) for c in categories]

# визуализируем
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=categories, cmap=plt.cm.get_cmap("jet", 6))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)









##########  K-means (get k value) ########## 

import gensim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# если в ipytho notebook, раскоментируйте следующую строчку
# %matplotlib inline
matplotlib.style.use('ggplot')
plt.rcParams['figure.figsize'] = (17, 10)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12

# читаем файл с токенизированными и лемматизированными текстами объявлений, разбиваем тексты на слова
texts = []
with open('textsMC.txt', 'r', encoding='utf-8') as f:
    for line in f:
        if not line.startswith('#'):
            texts.append([w for w in line.strip('\n').split()])
print(len(texts), np.mean([len(text) for text in texts]))

# векторизуем тексты
dictionary = gensim.corpora.Dictionary(texts)
print('Original: {}'.format(dictionary))
dictionary.filter_extremes(no_below = 5, no_above = 0.5, keep_n=None)
print('Filtered: {}'.format(dictionary))
corpus = [dictionary.doc2bow(text) for text in texts]

# строим матрицу tf-idf
tfidf = gensim.models.TfidfModel(corpus, normalize=True)
corpus_tfidf = tfidf[corpus]

# строим LSI модель
lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)

# кластеризуем, визуализируем и определяем оптимальное k 
fcoords = open("coords.csv", 'w')
for vector in lsi[corpus]:
    if len(vector) != 2:
        continue
    fcoords.write("%6.4f\t%6.4f\n" % (vector[0][1], vector[1][1]))
fcoords.close()
MAX_K = 20
X = np.loadtxt("coords.csv", delimiter="\t")
ks = range(1, MAX_K + 1)
inertias = np.zeros(MAX_K)
diff = np.zeros(MAX_K)
diff2 = np.zeros(MAX_K)
diff3 = np.zeros(MAX_K)
for k in ks:
    kmeans = KMeans(k).fit(X)
    inertias[k - 1] = kmeans.inertia_ 
    if k > 1:
        diff[k - 1] = inertias[k - 1] - inertias[k - 2]
    if k > 2:
        diff2[k - 1] = diff[k - 1] - diff[k - 2]
    if k > 3:
        diff3[k - 1] = diff2[k - 1] - diff2[k - 2]
elbow = np.argmin(diff3[3:]) + 3
plt.plot(ks, inertias, "b*-")
plt.plot(ks[elbow], inertias[elbow], marker='o', markersize=12,
         markeredgewidth=2, markeredgecolor='r', markerfacecolor=None)
plt.ylabel("Inertia")
plt.xlabel("K")

# кластеризуем с получившимся числом кластеров и визуализируем кластеры
NUM_TOPICS = 4  # ввести получившееся k
X = np.loadtxt("coords.csv", delimiter="\t")
kmeans = KMeans(NUM_TOPICS).fit(X)
y = kmeans.labels_
colors = ["b", "c", "r", "m"]
for i in range(X.shape[0]):
    plt.scatter(X[i][0], X[i][1], c=colors[y[i]], s=10)
Anna Marakasova
