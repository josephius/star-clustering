import numpy as np
'''
 classes for creating a matrix of the distances between all members of a set of vectors
'''


class Angular(object):

    @staticmethod
    def calc(vectors, center_cols=True, center_rows=True):
        n = vectors.shape[0]
        target = np.zeros((n, n), dtype='float32')
        if center_cols:
            vectors = vectors - np.expand_dims(np.average(vectors, axis=0), axis=0)
        if center_rows:
            vectors = vectors - np.expand_dims(np.average(vectors, axis=-1), axis=-1)

        # vector norms remain the same each loop iteration so only have to be calculated once
        norms = np.linalg.norm(vectors, axis=-1)
        # loop calculates cosine similarity between 1 vector and all vectors each iteration
        for i in range(n):
            target[i] = (vectors.dot(vectors[i])) / (norms[i] * norms)

        # convert cosine similarities to angular distance
        target = np.arccos(target) / np.pi
        return target


class Euclidean(object):

    @staticmethod
    def calc(vectors, center_cols=False, center_rows=False):
        n = vectors.shape[0]
        d = vectors.shape[1]
        target = np.zeros((n, n, d), dtype='float32')

        if center_cols:
            vectors = vectors - np.expand_dims(np.average(vectors, axis=0), axis=0)
        if center_rows:
            vectors = vectors - np.expand_dims(np.average(vectors, axis=-1), axis=-1)

        for i in range(n):
            for j in range(n):
                target[i, j] = vectors[i] - vectors[j]

        target = np.linalg.norm(target, axis=-1)
        return target
