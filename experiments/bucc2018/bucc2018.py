from metrics.utils.dataset import DatasetLoader
from metrics.contrastscore import ContrastScore
from collections import defaultdict
from tabulate import tabulate
import logging
import time
import datetime

logging.basicConfig(level=logging.INFO, datefmt="%m-%d %H:%M", format="%(asctime)s %(levelname)-8s %(message)s")

def train_contrastive(model, source_lang, target_lang, max_len, train_batch_size, num_epochs, iterations):
    dataset = DatasetLoader(source_lang, target_lang, max_monolingual_sent_len=max_len)
    mono_src, mono_tgt = dataset.load("monolingual-train")

    scorer = ContrastScore(model_name=model, source_language=source_lang, target_language=target_lang, parallelize=True, 
                           train_batch_size=train_batch_size, num_epochs=num_epochs)

    for iteration in range(1, iterations + 1):
        logging.info(f"Training iteration {iteration}.")
        scorer.suffix = f"{max_len}-{iteration}"
        scorer.train(mono_src, mono_tgt, overwrite=False)
    
    return scorer 


# Train a scorer or load a pretrained one
scorer = train_contrastive("sentence-transformers/distiluse-base-multilingual-cased-v2", "de", "en", max_len=32, 
                           train_batch_size=32, num_epochs=8, iterations=6)


# Load bucc datasets and create dicts that map sentences to its ids
f = open('de-en.training.de', "r")
lines = f.readlines()
sent2id_de = {}
for x in lines:
    sent2id_de[x.split('\t')[1][:-1]] = x.split('\t')[0]
f.close()

f = open('de-en.training.en', "r")
lines = f.readlines()
sent2id_en = {}
for x in lines:
    sent2id_en[x.split('\t')[1][:-1]] = x.split('\t')[0]
f.close()

# Extraxct sentences
sent_de = list(sent2id_de.keys())
sent_en = list(sent2id_en.keys())

# Apply unsupervised MT with ratio margin function to mine pseudo parallel data
bucc_candidates = scorer.mine(sent_de, sent_en, scorer.train_size, True)

# Calculate optimal threshold given the gold alignments
def BuccOptimize(candidate2score, gold):
    items = sorted(candidate2score.items(), key=lambda x: -x[1])
    ngold = len(gold)
    nextract = ncorrect = 0
    threshold = 0
    best_f1 = 0
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
    return threshold

# translate candidate pairs to its IDs 
candidate2score = {}
for line in bucc_candidates:
    score, src, trg = line.split('\t')
    #trg, src, score = line.split('\t')
    score = float(score)
    src = src.strip()
    trg = trg.strip()
    if src in sent2id_de and trg in sent2id_en:
        src_id = sent2id_de[src]
        trg_id = sent2id_en[trg]
        score = max(score, candidate2score.get((src_id, trg_id), score))
        candidate2score[(src_id, trg_id)] = score

# get gold alignments
gold = {line.strip() for line in open('de-en.training.gold', 'r')}
# calculate threshold
threshold = BuccOptimize(candidate2score, gold)

# function to extract candidates given the calculated threshold
def BuccExtract(cand2score, th, fname):
    if fname:
        of = open(fname, 'w')
    bitexts = []
    for (src, trg), score in cand2score.items():
        if score >= th:
            bitexts.append(src + '\t' + trg)
            if fname:
                of.write(src + '\t' + trg + '\n')
    if fname:
        of.close()
    return bitexts

# extract bitexts given the calculated threshold
bitexts = BuccExtract(candidate2score, threshold, './candidate_ids.txt')

# calculate F1-Score
ncorrect = len(gold.intersection(bitexts))
if ncorrect > 0:
    precision = ncorrect / len(bitexts)
    recall = ncorrect / len(gold)
    f1 = 2*precision*recall / (precision + recall)
else:
    precision = recall = f1 = 0

print(' - best threshold={:f}: precision={:.2f}, recall={:.2f}, F1={:.2f}'.format(threshold, 100*precision, 100*recall, 100*f1))