# import library
from metrics.utils.dataset import DatasetLoader
from metrics.contrastscore import ContrastScore

# import utils
from utils.bucc import load_bucc_dataset, bucc_optimize, bucc_extract
from utils.contrastive import train_contrastive

# import logging
import logging

# configure the logger
logging.basicConfig(level=logging.INFO, datefmt="%m-%d %H:%M", format="%(asctime)s %(levelname)-8s %(message)s")


# Train a scorer or load a pretrained one.
# For this experiment we use the "sentence-transformers/distiluse-base-multilingual-cased-v2" pretrained transformer
# model while training it from the source language "de" (German) to the target language "en" (English) with a max number
# of 50 words per sentence, while running 10 iterations with each 1 epoch and a batch size of 128 samples (compare to
# SimCSE paper for reference values).
scorer = train_contrastive("sentence-transformers/distiluse-base-multilingual-cased-v2", "de", "en", max_len=50,
                           train_batch_size=128, num_epochs=1, iterations=10)

# Load bucc datasets and create dicts that map sentences to its ids
sent2id_de = load_bucc_dataset("de-en.training.de")
sent2id_en = load_bucc_dataset("de-en.training.en")

# Extraxct sentences
sent_de = list(sent2id_de.keys())
sent_en = list(sent2id_en.keys())

# Apply unsupervised MT with ratio margin function to mine pseudo parallel data
bucc_candidates = scorer.mine(sent_de, sent_en, train_size=10000, overwrite=True)

# translate candidate pairs to its IDs
candidate2score = {}
for line in bucc_candidates:
    score, src, trg = line.split('\t')
    # trg, src, score = line.split('\t')
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

# calculate threshold. Not needed in our setting
# threshold = bucc_optimize(candidate2score, gold)

# extract bitexts given the calculated threshold
bitexts = bucc_extract(candidate2score, 0, None)

# calculate F1-Score
ncorrect = len(gold.intersection(bitexts))
if ncorrect > 0:
    precision = ncorrect / len(bitexts)
    recall = ncorrect / len(gold)
    f1 = 2 * precision * recall / (precision + recall)
else:
    precision = recall = f1 = 0

# print the final results
print(' - precision={:.2f}, recall={:.2f}, F1={:.2f}'.format(100 * precision, 100 * recall, 100 * f1))
