import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity


def relevance_feedback(vec_docs, vec_queries, sim, n=10):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """

    alpha = 0.67
    beta = 0.33
    rfsim = sim

    for iterations in range(3):
        for i in range(sim.shape[1]):
            print(i)
            relevant_documents = np.argsort(-rfsim[:, i])
            non_relevant_documents = np.argsort(rfsim[:, i])
            relevant = None
            non_relevant = None
            for idx, j in enumerate(relevant_documents):
                if idx < n:
                    if relevant is None:
                        relevant = vec_docs[j]
                    else:
                        relevant += vec_docs[j]
                else:
                    break
            for idx, j in enumerate(non_relevant_documents):
                if idx < n:
                    if non_relevant is None:
                        non_relevant = vec_docs[j]
                    else:
                        non_relevant += vec_docs[j]
                else:
                    break
            vec_queries[i] += (alpha * relevant) - (beta * non_relevant)
        rf_sim = cosine_similarity(vec_docs, vec_queries)
    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, queries, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    alpha = 0.67
    beta = 0.33
    rfsim = sim
    inverse_vocabulary = dict()
    vocabulary = dict()
    length_vocab = len(tfidf_model.vocabulary_.keys())
    vectorize_vocab = [0 for _ in range(length_vocab)]
    for key, value in tfidf_model.vocabulary_.items():
        inverse_vocabulary[value] = key
        vocabulary[key] = value
        vectorize_vocab[value] = key

    vectorize_vocab = tfidf_model.transform(vectorize_vocab)
    thesaurus = cosine_similarity(vectorize_vocab, vectorize_vocab)

    queries_1 = []

    for idx, query in enumerate(queries):
        queries_1.append(query.split())

    for iterations in range(3):
        for i in range(sim.shape[1]):
            print(i)
            relevant_documents = np.argsort(-rfsim[:, i])
            non_relevant_documents = np.argsort(rfsim[:, i])
            relevant = None
            non_relevant = None

            for idx, j in enumerate(relevant_documents):
                if idx < n:
                    if relevant is None:
                        relevant = vec_docs[j]
                    else:
                        relevant += vec_docs[j]
                else:
                    break
            for idx, j in enumerate(non_relevant_documents):
                if idx < n:
                    if non_relevant is None:
                        non_relevant = vec_docs[j]
                    else:
                        non_relevant += vec_docs[j]
                else:
                    break

            vec_queries[i] += (alpha * relevant) - (beta * non_relevant)

            for word in queries_1[i]:
                if word in vocabulary:
                    index = vocabulary[word]
                    relevant_words = np.argsort(-thesaurus[index, :])[:10]
                    for j in relevant_words:
                        vec_queries[i] += tfidf_model.transform([inverse_vocabulary[j]]) / (len(relevant_words))

        rf_sim = cosine_similarity(vec_docs, vec_queries)
    return rf_sim
