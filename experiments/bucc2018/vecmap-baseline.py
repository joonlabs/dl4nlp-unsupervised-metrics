from metrics.utils import embed
from metrics.utils.knn import ratio_margin_align
from nltk.metrics.distance import edit_distance
import torch


# Set your parameters. Warning: This script is quite demanding both for system and gpu RAM.
# Approx. 90 GB of system RAM are required. Since 16 GB of gpu RAM were not enough to handle 
# the code, we run it with cpu-resources. 
src_lang = 'de'
trg_lang = 'en'
device = 'cpu'


def get_bucc_sentences(lang, set='training', pair='de-en'):  
    filename = pair + '.' + set + '.' + lang  
    f = open(filename, 'r')
    lines = f.readlines()
    sent2id= {}
    id2sent = {}
    for x in lines:
        sent2id[x.split('\t')[1][:-1]] = x.split('\t')[0]
        id2sent[x.split('\t')[0]] = x.split('\t')[1][:-1]
    f.close()
    sentences = list(sent2id.keys())
    return sentences, sent2id, id2sent

def calculate_embeddings(sents, embeddings, mask):
    sent_embeddings = torch.empty(len(sents), embeddings.shape[-1])
    sent_embeddings = torch.sum(embeddings * mask, 1) / torch.sum(mask, 1)
    return sent_embeddings

def get_baseline_sentence_embeddings(src_lang=src_lang, trg_lang=trg_lang, device=device, set='training', pair='de-en', batch_size=5000):
    
    # Extract sentences from de-en bucc training corpora
    src_sents, _, _ = get_bucc_sentences(src_lang, set, pair)
    trg_sents, _, _ = get_bucc_sentences(trg_lang, set, pair)

    # Map monolingual tokens to cross lingual space
    src_dict, trg_dict = embed.map_multilingual_embeddings(src_lang, trg_lang, batch_size, device=device)

    # Get cross lingual embeddings with corresponding mask
    src_embeddings, _, _, src_mask = embed.vecmap_embed(src_sents, src_dict, src_lang)
    trg_embeddings, _, _, trg_mask = embed.vecmap_embed(trg_sents, trg_dict, trg_lang)
    
    # Calculate cross lingual embeddings 
    source_sent_embeddings = calculate_embeddings(src_sents, src_embeddings, src_mask)
    target_sent_embeddings = calculate_embeddings(trg_sents, trg_embeddings, trg_mask)

def vecmap_mine(source_sent_embeddings, target_sent_embeddings, device, out, k=5, batch_size=100000):
    pairs, scores = ratio_margin_align(source_sent_embeddings, target_sent_embeddings, k, batch_size, device)
    with open(out, "wb") as f:
        idx = 0
        check_duplicates_set = set()
        for score, (src, tgt) in sorted(zip(scores, pairs), key=lambda tup: tup[0], reverse=True):
            src_sent, tgt_sent = sent_de[src], sent_en[tgt]
            if tgt_sent not in check_duplicates_set and edit_distance(src_sent, tgt_sent) / max(len(src_sent), len(tgt_sent)) > 0.5:
                check_duplicates_set.add(tgt_sent)
                f.write(f"{score}\t{src_sent}\t{tgt_sent}\n".encode())
                idx += 1
                if idx >= len(scores):
                    break






