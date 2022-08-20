from metrics.utils.dataset import DatasetLoader
from metrics.contrastscore import ContrastScore
import logging


def train_contrastive(model, source_lang, target_lang, max_len, train_batch_size, num_epochs, iterations):
    """
    Trains a hugging face transformer model using the UScore method with contrastive learning approach.

    @param model: The name of the model that should be used
    @param source_lang: The source language. E.g. "de"
    @param target_lang: The target language. E.g. "en"
    @param max_len: The maximal token length per sentence (words per sentence)
    @param train_batch_size: The batch size used during training
    @param num_epochs: The number of epochs for training
    @param iterations: The number of iterations for training
    @return: ContrastScore
    """

    # load the dataset including source and target language from the news dataset
    dataset = DatasetLoader(source_lang, target_lang, max_monolingual_sent_len=max_len)

    # extract data (src) and targets (tgt) in monolingual form
    mono_src, mono_tgt = dataset.load("monolingual-train")

    # initiate the ConstrastScore based on the given parameters
    contrast_scorer = ContrastScore(model_name=model, source_language=source_lang, target_language=target_lang,
                                    parallelize=True, train_batch_size=train_batch_size, num_epochs=num_epochs)

    # start the training iterations
    for iteration in range(1, iterations + 1):
        # log the iteration
        logging.info(f"Training iteration {iteration}.")
        # set the scorer suffix based on the maximal token length per sentence and iteration
        contrast_scorer.suffix = f"{max_len}-{iteration}"
        # train the scorer
        contrast_scorer.train(mono_src, mono_tgt, overwrite=False)

    # return the scorer object for later use
    return contrast_scorer