from joblib import Parallel, delayed
from textblob import TextBlob
from textblob.translate import NotTranslated

import argparse
import os
import pandas as pd

NAN_WORD = "_NAN_"

def translate(comment, language):
    try:
        if hasattr(comment, "decode"):
            comment = comment.decode("utf-8")

        text = TextBlob(comment)
        try:
            text = text.translate(to=language)
            text = text.translate(to="en")
        except NotTranslated:
            pass

        return str(text)
    except:
        return comment

train_file_path = "/extended_data/train_preprocessed_old.csv"
languages = ["de", "fr", "it", "ar", "es", "ja", "hi", "ru"]
thread_count = 1000

train_data = pd.read_csv(train_file_path)
comments_list = train_data["comment_text"].fillna(NAN_WORD).values

parallel = Parallel(thread_count, backend="threading", verbose=5)
for language in languages:
    print('Translate comments using "{0}" language'.format(language))
    translated_data = parallel(delayed(translate)(comment, language) for comment in comments_list)
    train_data["comment_text"] = translated_data

    result_path = "/output/" + "train_extended_preprocessed_" + language + ".csv"
    train_data.to_csv(result_path, index=False)
