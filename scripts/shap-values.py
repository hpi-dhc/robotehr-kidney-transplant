#!/usr/bin/env python

import pickle
import sys

import matplotlib.pyplot as plt
import shap

def main(use_case):
    results = pickle.load(open(f'/home/martet02/shap-{use_case}.pickle', 'rb'))

    shap_values = results['shap_values']
    X_test = results['X_test']

    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = [r'\usepackage[cm]{sfmath}']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'cm'
    plt.rcParams['font.size'] = 14


    shap.summary_plot(shap_values[1], X_test, max_display=10, show=False)

    f = plt.gcf()

    f.savefig(f'/home/martet02/shap-{use_case}.pdf', bbox_inches="tight")
    f.savefig(f'/home/martet02/shap-{use_case}.png', bbox_inches="tight", dpi=300)

    print("Done")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("Usage: python shap-values.py {uti|cmv|ovi}")
