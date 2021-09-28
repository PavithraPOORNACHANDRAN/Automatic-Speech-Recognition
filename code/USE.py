from tempfile import TemporaryFile
from absl import logging
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import csv
import seaborn as sns

module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
# Import the Universal Sentence Encoder's TF Hub module
model = hub.load(module_url)


# Load the data
data = pd.read_csv("useinput.csv")
data.columns =  ['sent_id','sentence']
data.head()

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_hub as hub
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)


session = tf.Session()
tf.keras.backend.set_session(session)
session.run(tf.global_variables_initializer())
session.run(tf.tables_initializer())
message_embeddings = session.run(embed(data["sentence"].values.tolist()))
print(message_embeddings.shape)
print(type(message_embeddings))



with open('useoutput8.csv', 'w', newline='') as file:
        writer = csv.writer(file,quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(["Id", "Embedding"])
        for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
                #print("Message: {}".format(data.sent_id[i]))
                #print("Embedding size: {}".format(len(message_embedding)))
                message_embedding_snippet = ", ".join((str(x) for x in message_embedding[:]))
                #print("Embedding[{},...]\n".
                # format(message_embedding_snippet))
                line="["+ format(message_embedding_snippet)+"]"

                writer.writerow([data.sent_id[i], line])
