from metrics.utils.dataset import DatasetLoader
from metrics.contrastscore import ContrastScore
import logging


def load_bucc_dataset(dataset_file):
    """
    Loads a BUCC dataset from a file and returns the data in a dictionary.

    @param dataset_file:
    @return:
    """
    # init the data
    data = {}

    # open the dataset
    dataset = open(dataset_file, "r")

    # read the lines
    dataset_lines = dataset.readlines()

    # iterate over all lines and add data to data object
    for entry in dataset_lines:
        data[entry.split('\t')[1][:-1]] = entry.split('\t')[0]

    # close file resource
    dataset.close()

    # return data
    return data


def bucc_optimize(candidate_2_score, gold):
    """
    Calculate optimal threshold given the gold alignments. This function is used,
    when the test set gold alignments are available. We will use it anyway to create
    candidates, settung threshold to 0 later.

    @param candidate_2_score:
    @param gold:
    @return:
    """
    # get all items
    items = sorted(candidate_2_score.items(), key=lambda x: -x[1])

    # get the number of gold
    ngold = len(gold)

    # inits number of extracted and of correct to be zero
    nextract = ncorrect = 0

    # init threshold and best f1 score also to be zero
    threshold = 0
    best_f1 = 0

    # iterate over all items
    for i in range(len(items)):
        nextract += 1
        if '\t'.join(items[i][0]) in gold:
            ncorrect += 1
        if ncorrect > 0:
            precision = ncorrect / nextract
            recall = ncorrect / ngold
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                threshold = (items[i][1] + items[i + 1][1]) / 2

    # return the threshold
    return threshold


def bucc_extract(cand_2_score, th, fname):
    """
    Extract the candidates given a threshold.

    @param cand_2_score:
    @param th:
    @param fname:
    @return:
    """

    # check if should be written to file
    if fname:
        of = open(fname, 'w')

    # init bitexts as empty array
    bitexts = []

    # iterate over all cand_2_score items
    for (src, trg), score in cand_2_score.items():
        if score >= th:
            bitexts.append(src + '\t' + trg)

            # if should be written, append source and target to file
            if fname:
                of.write(src + '\t' + trg + '\n')

    # close resource if should be written
    if fname:
        of.close()

    # return the bitexts
    return bitexts