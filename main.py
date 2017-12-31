'''EM algorithm.
Author: Yusuke Kanai
mail: kanayusu@gmail.com

This program is a sample of EM algorithm using MNIST
'''
from __future__ import print_function

from scipy.stats import multivariate_normal
import numpy as np
import mnist
from sklearn import decomposition
from matplotlib import pyplot


class EMAlgorithm(object):
    def __init__(self):
        self.samples = 2000
        self.pca_dim = 50
        self.pca = decomposition.PCA(n_components=self.pca_dim)

        mnist_data = mnist.train_images().reshape(-1, 784) / 255.
        self.image_data = mnist_data[0:self.samples]
        self.pca.fit(self.image_data)
        self.image_data = self.pca.transform(self.image_data)

        # mixing coefficient
        self.mc = [0.1] * 10
        self.comp_mean = [np.random.random(self.pca_dim) for i in range(10)]
        self.comp_var = [np.identity(self.pca_dim, dtype='f')] * 10

    def e_step(self, datas):
        gen_prob = []

        for i in range(10):
            gen_prob_k = self.mc[i] * multivariate_normal.pdf(datas, mean=self.comp_mean[i], cov=self.comp_var[i],
                                                              allow_singular=True)
            gen_prob.append(gen_prob_k)

        gen_prob = gen_prob / sum(gen_prob)
        return gen_prob

    def m_step(self, datas, brob):
        nk = np.sum(brob, axis=1)
        for i in range(10):
            self.comp_mean[i] = datas.transpose().dot(brob[i]) / nk[i]
        for i in range(10):
            diff = datas - self.comp_mean[i]
            self.comp_var[i] = diff.transpose().dot(np.multiply(diff, brob[i].reshape((self.samples, 1)))) / nk[i]
        self.mc = nk / datas.shape[0]

    def log_likelihood(self, datas):
        res = 0.
        for data in datas:
            res_i = 0.
            for i in range(10):
                res_i += self.mc[i] * multivariate_normal.pdf(data, mean=self.comp_mean[i], cov=self.comp_var[i],
                                                              allow_singular=True)
            res += np.log(res_i)
        return res

    def print_label(self, name):
        result = []
        for i in range(10):
            result.append(multivariate_normal.rvs(mean=self.comp_mean[i], cov=self.comp_var[i], size=10))
        result = self.pca.inverse_transform(np.stack(result, axis=0)).reshape(10, 10, 28, 28)
        output = 255 * result.transpose(0, 2, 1, 3).reshape((10 * 28, 10 * 28))
        pyplot.imshow(output)
        pyplot.savefig(name)

    def train(self):
        epoch = 20
        for i in range(epoch):
            prob = self.e_step(self.image_data)
            self.m_step(self.image_data, prob)
            print(self.log_likelihood(self.image_data))
            self.print_label("sample_{}".format(i))


def main():
    model = EMAlgorithm()
    model.train()


if __name__ == '__main__':
    main()
