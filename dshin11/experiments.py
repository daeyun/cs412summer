import sklearn
from dshin import plot_utils
import sklearn.mixture
import sklearn.ensemble
import wordcloud
import pickle
import matplotlib.pyplot as pt
import os
from sklearn import metrics
import numpy as np
from os import path
from os.path import join
from sklearn.decomposition import PCA
from dshin import geom3d
from sklearn.feature_extraction.text import TfidfTransformer

data_ = None


def tfidf(cluster_counts):
    transformer = TfidfTransformer(norm=None, sublinear_tf=False, use_idf=True, smooth_idf=True)
    out = transformer.fit_transform(cluster_counts)
    return out.toarray()


def display_qualitative(k, cluster_distances, dimension):
    # display qualitative results
    labels = np.argmin(cluster_distances, axis=1)
    for ii in range(k):
        inds = np.argsort(cluster_distances[:, ii])
        top_papers = [data_['papers'][i] for i in inds[:5]]
        for i, name in enumerate(top_papers):
            item = data_['metadata']['{}'.format(name)]
            print(item['title'])
        print()
    cluster_counts = []
    for ii in range(k):
        counts = data_['tables']['paper'][dimension][np.array(labels) == ii].copy().astype(np.float64)
        # counts *= cluster_distances[np.array(labels) == ii].min(axis=1)[:,None]
        counts = counts.sum(axis=0)
        cluster_counts.append(counts)
    cluster_counts = np.array(cluster_counts).astype(np.float64)

    out = tfidf(cluster_counts)
    for ii in range(k):
        words = [data_['words'][dimension][i] for i in np.argsort(out[ii])[::-1][:5]]
        print(', '.join([item.split(',')[0] for item in words]))

        cloud = wordcloud.WordCloud().generate_from_frequencies(
            {data_['words'][dimension][i].split(',')[0]: val + 0.5 for i, val in enumerate(out[ii])}
        )
        plot_utils.set_figsize(8, 8)
        pt.imshow(cloud, interpolation='bilinear')
        pt.axis('off')
        pt.show()
    plot_utils.set_figsize(6, 6)


def clustering(dimension='method', k=4, threshold=8):
    plot_utils.set_figsize(6, 6)
    proj_root = path.realpath(join(path.dirname(path.realpath(__file__)), '..'))

    global data_
    if data_ is None:
        with open(join(proj_root, 'pkl', 'all.pkl'), 'rb') as f:
            data = pickle.load(f)
        data_ = data
    else:
        data = data_

    X = data['tables']['paper'][dimension].copy()
    X = (X > threshold).astype(np.float64)

    # whitening
    # s = X.std(axis=1, keepdims=True)
    # X[s.ravel()>0] /= s[s>0,None]


    pca2 = PCA(n_components=2, whiten=True)
    out2 = pca2.fit_transform(X)
    pca3 = PCA(n_components=3)
    out3 = pca3.fit_transform(X)

    cl = sklearn.cluster.KMeans(n_clusters=k, max_iter=2000, n_init=800)
    t = cl.fit_transform(X)
    labels = np.argmin(t, axis=1)
    assert (labels == cl.labels_).all()
    score = metrics.silhouette_score(X, labels, metric='euclidean')

    print('silhouette score = ', score)

    display_qualitative(k, t, dimension)

    pt.figure()

    colors = ['red', 'blue', 'purple', 'green', 'orange', 'yellow', 'cyan', 'm']
    for i in range(k):
        pt.scatter(out2[labels == i, 0], out2[labels == i, 1], marker='.', c=colors[i])
        pt.axis('square')

    axis = None
    for i in range(k):
        ax = geom3d.pts(out3[labels == i], color=colors[i], ax=axis, reset_limits=False, markersize=10)
        if axis is None:
            axis = ax

    ##
    pt.show()
    print('----')

    X = data['tables']['paper'][dimension].copy()

    print(X)

    for mode in [0, 1]:
        if mode == 0:
            X = data['tables']['paper'][dimension].copy()
            # whitening
            m = (X - X.mean(axis=1, keepdims=True))
            s = X.std(axis=1, keepdims=True)
            m[s.ravel() > 0] /= s[s > 0, None]
            X = m
        elif mode == 1:
            X = data['tables']['paper'][dimension].copy()

        scores = []
        for kk in range(2, 11):
            cl = sklearn.cluster.KMeans(n_clusters=kk, max_iter=1000, n_init=50)
            t = cl.fit_transform(X)
            labels = cl.labels_
            score = metrics.silhouette_score(X, labels, metric='euclidean')
            scores.append(score)
        print('mode:', mode)
        pt.figure()
        pt.plot(list(range(2, len(scores) + 2)), scores, marker='o')
        pt.xlabel('number of clusters')
        pt.ylabel('silhouette coefficient')
        pt.show()
    ###






    pt.show()
    print('----')

    X = data['tables']['paper'][dimension].copy()
    # whiten
    m = (X - X.mean(axis=1, keepdims=True))
    s = X.std(axis=1, keepdims=True)
    m[s.ravel() > 0] /= s[s > 0, None]
    X = m

    pca2 = PCA(n_components=2)
    out2 = pca2.fit_transform(X)
    pca3 = PCA(n_components=3)
    out3 = pca3.fit_transform(X)

    # dont whiten for kmeans
    X = data['tables']['paper'][dimension].copy()

    cl = sklearn.cluster.KMeans(n_clusters=k, max_iter=1000, n_init=1000)
    t = cl.fit_transform(X)
    print()

    display_qualitative(k, t, dimension)

    labels = cl.labels_
    score = metrics.silhouette_score(X, labels, metric='euclidean')
    print('silhouette score = ', score)

    colors = ['red', 'blue', 'purple', 'green', 'orange', 'yellow', 'cyan', 'm']
    pt.figure()
    for i in range(k):
        pt.scatter(out2[labels == i, 0], out2[labels == i, 1], marker='.', c=colors[i])
        pt.axis('square')

    axis = None
    for i in range(k):
        ax = geom3d.pts(out3[labels == i], color=colors[i], ax=axis, reset_limits=False, markersize=10)
        if axis is None:
            axis = ax

    pt.show()
    print('---- whitening')

    X = data['tables']['paper'][dimension].copy()
    # whiten
    m = (X - X.mean(axis=1, keepdims=True))
    s = X.std(axis=1, keepdims=True)
    m[s.ravel() > 0] /= s[s > 0, None]
    X = m

    pca2 = PCA(n_components=2)
    out2 = pca2.fit_transform(X)
    pca3 = PCA(n_components=3)
    out3 = pca3.fit_transform(X)

    cl = sklearn.cluster.KMeans(n_clusters=k, max_iter=1000, n_init=500)
    t = cl.fit_transform(X)
    labels = np.argmin(t, axis=1)
    print()

    display_qualitative(k, t, dimension)

    score = metrics.silhouette_score(X, labels, metric='euclidean')
    print('silhouette score = ', score)

    colors = ['red', 'blue', 'purple', 'green', 'orange', 'yellow', 'cyan', 'm']
    pt.figure()
    for i in range(k):
        pt.scatter(out2[labels == i, 0], out2[labels == i, 1], marker='.', c=colors[i])
        pt.axis('square')

    axis = None
    for i in range(k):
        ax = geom3d.pts(out3[labels == i], color=colors[i], ax=axis, reset_limits=False, markersize=10)
        if axis is None:
            axis = ax

    pt.show()
    print('-------')

    X = data['tables']['paper'][dimension].copy()
    # whiten
    m = (X - X.mean(axis=1, keepdims=True))
    s = X.std(axis=1, keepdims=True)
    m[s.ravel() > 0] /= s[s > 0, None]
    X = m

    cl = sklearn.mixture.GaussianMixture(k)
    cl.fit(X)
    labels = cl.predict(X)
    print()

    t = cl.predict_proba(X)

    display_qualitative(k, 1.0 / (t + 0.4), dimension)

    pt.figure()

    colors = ['red', 'blue', 'purple', 'green', 'orange', 'yellow', 'cyan', 'm']
    for i in range(k):
        pt.scatter(out2[labels == i, 0], out2[labels == i, 1], marker='.', c=colors[i])
        pt.axis('square')

    axis = None
    for i in range(k):
        ax = geom3d.pts(out3[labels == i], color=colors[i], ax=axis, reset_limits=False, markersize=10)
        if axis is None:
            axis = ax


from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict


def classification():
    plot_utils.set_figsize(6, 6)
    proj_root = path.realpath(join(path.dirname(path.realpath(__file__)), '..'))

    global data_
    if data_ is None:
        with open(join(proj_root, 'pkl', 'all.pkl'), 'rb') as f:
            data = pickle.load(f)
        data_ = data
    else:
        data = data_

    citations = np.array([data['metadata'][item]['citation_count'] for item in data['papers']])

    X = np.concatenate((data['tables']['paper']['method'],data['tables']['paper']['problem'],data['tables']['paper']['author']),axis=1)
    # X = data['tables']['paper']['author']
    y = citations > 0

    inds = np.arange(len(citations))
    np.random.shuffle(inds)

    clfs = [
        sklearn.ensemble.RandomForestClassifier(n_estimators=500, min_samples_split=2),
        sklearn.neighbors.KNeighborsClassifier(5),
        sklearn.svm.SVC(kernel='rbf', gamma=10, C=1),
        sklearn.svm.SVC(kernel="linear", C=1),
        sklearn.ensemble.AdaBoostClassifier(n_estimators=50)
        # sklearn.gaussian_process.GaussianProcessClassifier(1.0 * sklearn.gaussian_process.kernels.RBF(1.0), warm_start=True),
    ]

    for clf in clfs:
        out = cross_val_predict(clf, X, y, cv=10, n_jobs=6)
        print(clf.__class__)
        score = metrics.accuracy_score(y, out)
        print(score)
    return citations


import orangecontrib


def association():
    plot_utils.set_figsize(6, 6)
    proj_root = path.realpath(join(path.dirname(path.realpath(__file__)), '..'))

    global data_
    if data_ is None:
        with open(join(proj_root, 'pkl', 'all.pkl'), 'rb') as f:
            data = pickle.load(f)
        data_ = data
    else:
        data = data_

    print("1. problem -> method")
    freq = orangecontrib.associate.frequent_itemsets(np.concatenate((data['tables']['paper']['problem'], data['tables']['paper']['method']), axis=1), min_support=0.02)
    itemset = {k: v for k, v in freq}
    print(len(itemset))

    rules = list(orangecontrib.associate.association_rules(itemset, 0.7))
    rr = [item for item in rules]
    print(len(rr))

    range1 = set(np.arange(data['tables']['paper']['problem'].shape[1]))
    range2 = set(np.arange(data['tables']['paper']['method'].shape[1]) + data['tables']['paper']['problem'].shape[1])

    results = []
    for l, r, s1, s2 in rr:
        if l.issubset(range1) and r.issubset(range2) and len(r) < 7:
            results.append((l, r, s1, s2))
    print(len(results))

    print('#####')

    stats = list(orangecontrib.associate.rules_stats(results, itemset, data['tables']['paper']['problem'].shape[0]))

    for nn, mm in (('support', 2), ('confidence', 3), ('lift', 6)):
        print('Sorted by {}'.format(nn))
        if nn == 'confidence':
            inds = np.argsort([(item[mm] * 100000 + item[6]) for item in stats])[::-1]
            print(inds)
        else:
            inds = np.argsort([item[mm] for item in stats])[::-1]
        for j in range(20):
            i = inds[j]
            lt = ', '.join([data['words']['problem'][ii].split(',')[0] for ii in results[i][0]])
            rt = ', '.join([data['words']['method'][ii - len(data['words']['problem'])].split(',')[0] for ii in results[i][1]])
            item = stats[i]
            line_r = '{}, {:.2f}, {:.2f}   {{{}}} -> {{{}}}'.format(item[2], item[3], item[6], lt, rt)
            print(line_r)

    print('#####')
    print('#####')

    print("2. problem -> method: singleton problems")
    freq = orangecontrib.associate.frequent_itemsets(np.concatenate((data['tables']['paper']['problem'], data['tables']['paper']['method']), axis=1), min_support=0.02)
    itemset = {k: v for k, v in freq}
    print(len(itemset))

    rules = list(orangecontrib.associate.association_rules(itemset, 0.7))
    rr = [item for item in rules]
    print(len(rr))

    range1 = set(np.arange(data['tables']['paper']['problem'].shape[1]))
    range2 = set(np.arange(data['tables']['paper']['method'].shape[1]) + data['tables']['paper']['problem'].shape[1])

    results = []
    for l, r, s1, s2 in rr:
        if l.issubset(range1) and r.issubset(range2) and len(l) == 1 and len(r) < 7:
            results.append((l, r, s1, s2))
    print(len(results))

    print('#####')

    stats = list(orangecontrib.associate.rules_stats(results, itemset, data['tables']['paper']['problem'].shape[0]))

    for nn, mm in (('support', 2), ('confidence', 3), ('lift', 6)):
        print('Sorted by {}'.format(nn))
        if nn == 'confidence':
            inds = np.argsort([(item[mm] * 100000 + item[6]) for item in stats])[::-1]
        else:
            inds = np.argsort([item[mm] for item in stats])[::-1]
        for j in range(20):
            i = inds[j]
            lt = ', '.join([data['words']['problem'][ii].split(',')[0] for ii in results[i][0]])
            rt = ', '.join([data['words']['method'][ii - len(data['words']['problem'])].split(',')[0] for ii in results[i][1]])
            item = stats[i]
            line_r = '{}, {:.2f}, {:.2f}   {{{}}} -> {{{}}}'.format(item[2], item[3], item[6], lt, rt)
            print(line_r)

    print('#####')
    print('#####')

    # 3. method -> problem

    print("3. method -> problem")
    freq = orangecontrib.associate.frequent_itemsets(np.concatenate((data['tables']['paper']['method'], data['tables']['paper']['problem']), axis=1), min_support=0.02)
    itemset = {k: v for k, v in freq}
    print(len(itemset))

    rules = list(orangecontrib.associate.association_rules(itemset, 0.7))
    rr = [item for item in rules]
    print(len(rr))

    range1 = set(np.arange(data['tables']['paper']['method'].shape[1]))
    range2 = set(np.arange(data['tables']['paper']['problem'].shape[1]) + data['tables']['paper']['method'].shape[1])

    results = []
    for l, r, s1, s2 in rr:
        if l.issubset(range1) and r.issubset(range2) and len(r) < 7:
            results.append((l, r, s1, s2))
    print(len(results))

    print('#####')

    stats = list(orangecontrib.associate.rules_stats(results, itemset, data['tables']['paper']['method'].shape[0]))

    for nn, mm in (('support', 2), ('confidence', 3), ('lift', 6)):
        print('Sorted by {}'.format(nn))
        if nn == 'confidence':
            inds = np.argsort([(item[mm] * 100000 + item[6]) for item in stats])[::-1]
        else:
            inds = np.argsort([item[mm] for item in stats])[::-1]
        for j in range(20):
            i = inds[j]
            lt = ', '.join([data['words']['method'][ii].split(',')[0] for ii in results[i][0]])
            rt = ', '.join([data['words']['problem'][ii - len(data['words']['method'])].split(',')[0] for ii in results[i][1]])
            item = stats[i]
            line_r = '{}, {:.2f}, {:.2f}   {{{}}} -> {{{}}}'.format(item[2], item[3], item[6], lt, rt)
            print(line_r)

    print('#####')
    print('#####')

    # 4. method -> problem: singleton methods
    print("4. method -> problem: singleton methods")
    freq = orangecontrib.associate.frequent_itemsets(np.concatenate((data['tables']['paper']['method'], data['tables']['paper']['problem']), axis=1), min_support=0.02)
    itemset = {k: v for k, v in freq}
    print(len(itemset))

    rules = list(orangecontrib.associate.association_rules(itemset, 0.7))
    rr = [item for item in rules]
    print(len(rr))

    range1 = set(np.arange(data['tables']['paper']['method'].shape[1]))
    range2 = set(np.arange(data['tables']['paper']['problem'].shape[1]) + data['tables']['paper']['method'].shape[1])

    results = []
    for l, r, s1, s2 in rr:
        if l.issubset(range1) and r.issubset(range2) and len(l) == 1 and len(r) < 7:
            results.append((l, r, s1, s2))
    print(len(results))

    print('#####')

    stats = list(orangecontrib.associate.rules_stats(results, itemset, data['tables']['paper']['method'].shape[0]))

    for nn, mm in (('support', 2), ('confidence', 3), ('lift', 6)):
        print('Sorted by {}'.format(nn))
        if nn == 'confidence':
            inds = np.argsort([(item[mm] * 100000 + item[6]) for item in stats])[::-1]
        else:
            inds = np.argsort([item[mm] for item in stats])[::-1]
        for j in range(20):
            i = inds[j]
            lt = ', '.join([data['words']['method'][ii].split(',')[0] for ii in results[i][0]])
            rt = ', '.join([data['words']['problem'][ii - len(data['words']['method'])].split(',')[0] for ii in results[i][1]])
            item = stats[i]
            line_r = '{}, {:.2f}, {:.2f}   {{{}}} -> {{{}}}'.format(item[2], item[3], item[6], lt, rt)
            print(line_r)

    print('#####')
    print('#####')

    print('#####')
    print('#####')
    print('#####')
    print('#####')
    print('#####')
    print('#####')

    print('5. author -> problem')

    freq = orangecontrib.associate.frequent_itemsets(np.concatenate((data['tables']['paper']['author'], data['tables']['paper']['problem']), axis=1), min_support=0.02)
    itemset = {k: v for k, v in freq}
    print(len(itemset))

    rules = list(orangecontrib.associate.association_rules(itemset, 0.8))
    rr = [item for item in rules]
    print(len(rr))

    range1 = set(np.arange(data['tables']['paper']['author'].shape[1]))
    range2 = set(np.arange(data['tables']['paper']['problem'].shape[1]) + data['tables']['paper']['author'].shape[1])

    results = []
    for l, r, s1, s2 in rr:
        if l.issubset(range1) and r.issubset(range2):
            results.append((l, r, s1, s2))
    print(len(results))

    stats = list(orangecontrib.associate.rules_stats(results, itemset, data['tables']['paper']['method'].shape[0]))

    for nn, mm in (('support', 2), ('confidence', 3), ('lift', 6)):
        print('Sorted by {}'.format(nn))
        if nn == 'confidence':
            inds = np.argsort([(item[mm]*10000 + item[2]) for item in stats])[::-1]
        else:
            inds = np.argsort([item[mm] for item in stats])[::-1]
        for j in range(len(inds)):
            i = inds[j]
            lt = ', '.join([data['words']['author'][ii].split(',')[0] for ii in results[i][0]])
            rt = ', '.join([data['words']['problem'][ii - len(data['words']['author'])].split(',')[0] for ii in results[i][1]])
            item = stats[i]
            line_r = '{}, {:.2f}, {:.2f}   {{{}}} -> {{{}}}'.format(item[2], item[3], item[6], lt, rt)
            print(line_r)

    print('#####')
    print('#####')
    print('#####')
    print('#####')

    print('6. problem -> author')

    freq = orangecontrib.associate.frequent_itemsets(np.concatenate((data['tables']['paper']['problem'], data['tables']['paper']['author']), axis=1), min_support=0.01)
    itemset = {k: v for k, v in freq}
    print(len(itemset))

    rules = list(orangecontrib.associate.association_rules(itemset, 0.6))
    rr = [item for item in rules]
    print(len(rr))

    range1 = set(np.arange(data['tables']['paper']['problem'].shape[1]))
    range2 = set(np.arange(data['tables']['paper']['author'].shape[1]) + data['tables']['paper']['problem'].shape[1])

    results = []
    for l, r, s1, s2 in rr:
        if l.issubset(range1) and r.issubset(range2):
            results.append((l, r, s1, s2))
    print(len(results))

    stats = list(orangecontrib.associate.rules_stats(results, itemset, data['tables']['paper']['method'].shape[0]))

    for nn, mm in (('support', 2), ('confidence', 3), ('lift', 6)):
        print('Sorted by {}'.format(nn))
        if nn == 'confidence':
            inds = np.argsort([(item[mm]*10000 + item[2]) for item in stats])[::-1]
        else:
            inds = np.argsort([item[mm] for item in stats])[::-1]
        for j in range(len(inds)):
            i = inds[j]
            lt = ', '.join([data['words']['problem'][ii].split(',')[0] for ii in results[i][0]])
            rt = ', '.join([data['words']['author'][ii - len(data['words']['problem'])].split(',')[0] for ii in results[i][1]])
            item = stats[i]
            line_r = '{}, {:.2f}, {:.2f}   {{{}}} -> {{{}}}'.format(item[2], item[3], item[6], lt, rt)
            print(line_r)
