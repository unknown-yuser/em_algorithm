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
        self.cls_num = 100
        self.pca = decomposition.PCA(n_components=30)

        mnist_data = mnist.train_images().reshape(-1, 784) / 255.
        mnist_ks = []
        self.label_data = mnist.train_labels()

        for i in range(10):
            mnist_ks.append(mnist_data[self.label_data == i][0:self.cls_num])
        self.image_data = np.concatenate(mnist_ks)

        self.pca.fit(self.image_data)
        self.image_data = self.pca.transform(self.image_data)

        # mixing coefficient
        self.mc = [0.1] * 10
        self.init_component(self.image_data)

    def init_component(self, images):
        c = []
        m = []
        v = []
        for i in range(10):
            c.append(images[i * self.cls_num:(i + 1) * self.cls_num])

        for i in range(10):
            m.append(np.mean(c[i], axis=0))
            vi = np.cov(c[i], rowvar=0, bias=1)
            v.append(vi)

        self.comp_mean = m
        self.comp_var = v

    def e_step(self, datas):
        gen_prob = []

        for i in range(10):
            gen_prob_k = self.mc[i] * multivariate_normal.pdf(datas, mean=self.comp_mean[i], cov=self.comp_var[i])
            gen_prob.append(gen_prob_k)

        gen_prob = gen_prob / sum(gen_prob)
        return gen_prob

    def m_step(self, datas, brob):
        nk = np.sum(brob, axis=1)
        for i in range(10):
            self.comp_mean[i] = datas.transpose().dot(brob[i]) / nk[i]
        for i in range(10):
            diff = datas - self.comp_mean[i]
            self.comp_var[i] = diff.transpose().dot(np.multiply(diff, brob[i].reshape((10*self.cls_num, 1)))) / nk[i]
        self.mc = nk / datas.shape[0]

    def log_likelihood(self, datas):
        res = 0.
        for data in datas:
            res_i = 0.
            for i in range(10):
                res_i += self.mc[i] * multivariate_normal.pdf(data, mean=self.comp_mean[i], cov=self.comp_var[i])
            res += np.log(res_i)
        return res

    def print_label(self, name, k):
        output = multivariate_normal.rvs(mean=self.comp_mean[k], cov=self.comp_var[k])
        output = 255 * self.pca.inverse_transform(output).reshape((28, 28))
        pyplot.imshow(output)
        pyplot.savefig(name)

    def train(self):
        epoch = 10
        for i in range(epoch):
            prob = self.e_step(self.image_data)
            self.m_step(self.image_data, prob)
            print(self.log_likelihood(self.image_data))
            self.print_label("sample_{}".format(i), i)


def main():
    model = EMAlgorithm()
    model.train()


if __name__ == '__main__':
    main()
