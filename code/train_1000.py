import re
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
import sys
import csv
from csv import writer
import os
import h5py

def read_csv(file):
    data=list()
    with open(file, 'r') as inp:
        for row in csv.reader(inp):
            data.append(row)
    return data

def str_process(instring):
    instring = re.sub("\[","",instring)
    instring = re.sub("\]","",instring)
    instring = re.sub(" ","",instring)
    float_vector = instring.split(",")
    float_vector = list(map(float,float_vector))
    return np.array(float_vector)

outputname=str(sys.argv[1])
max_length = 256  # Maximum length of input sentence to the model.
batch_size = 16
epochs = 2

# Labels in our dataset.
labels = ["0", "1" ]

train_df = pd.read_csv("train1000_bertkeras_aclm_cmpwithbesthypo_idsent.csv")
valid_df = pd.read_csv("dev_bertkeras_aclm_cmpwithbesthypo_idsent.csv")
test_df = pd.read_csv("train1000_bertkeras_aclm_cmpwithbesthypo_idsent.csv")

# Read as a list
train_df2 = read_csv("train1000_bertkeras_aclm_cmpwithbesthypo_idsent.csv")
train_id_previd = read_csv("train_id_1000.csv")
train_df_use = read_csv("use_output_1000.csv")


valid_df2 = read_csv("dev_bertkeras_aclm_cmpwithbesthypo_idsent.csv")
valid_id_previd = read_csv("dev_input_ids_8loc.csv")
valid_df_use = read_csv("dev_useoutput.csv")


test_df2 = read_csv("train1000_bertkeras_aclm_cmpwithbesthypo_idsent.csv")
test_id_previd = read_csv("train_id_1000.csv")
test_df_use = read_csv("use_output_1000.csv")



def add_embeddings(data_frame,pairs_df,id_previd,df_use):
    embed=list()
    for row in range(1, len(pairs_df)):
        for row1 in range(1, len(id_previd)):
            if pairs_df[row][7] == id_previd[row1][0]:
                id = id_previd[row1][1]
                for row2 in range(1, len(df_use)):
                    if id == df_use[row2][0]:
                        embed.append(df_use[row2][1])
    #print(len(pairs_df),len(id_previd), len(df_use), len(embed))
    data_frame['embedding']=embed
    # end of the function add embedding


add_embeddings(train_df,train_df2, train_id_previd, train_df_use)
add_embeddings(valid_df,valid_df2, valid_id_previd, valid_df_use)
add_embeddings(test_df, test_df2, test_id_previd, test_df_use)

emb_train = train_df['embedding'].apply(str_process)
emb_valid = valid_df['embedding'].apply(str_process)
emb_test = test_df['embedding'].apply(str_process)

# Shape of the data
#print(f"Total train samples : {train_df.shape[0]}")
#print(f"Total validation samples: {valid_df.shape[0]}")
#print(f"Total test samples: {valid_df.shape[0]}")

"""
Let's look at one sample from the dataset:
"""
#print(f"Sentence1: {train_df.loc[1, 'sentence1']}")
#print(f"Sentence2: {train_df.loc[1, 'sentence2']}")
#print(f"Similarity: {train_df.loc[1, 'similarity']}")

# We have some NaN entries in our train data, we will simply drop them.
print("Number of missing values")
print(train_df.isnull().sum())
train_df.dropna(axis=0, inplace=True)

"""
One-hot encode training, validation, and test labels.
"""

train_df["label"] = train_df["similarity"].apply(
        lambda x: 0 if x == 0 else 1 if x == 1 else 5
    )
y_train = tf.keras.utils.to_categorical(train_df.label, num_classes=2)

valid_df["label"] = valid_df["similarity"].apply(
        lambda x: 0 if x == 0 else 1 if x == 1 else 2
    )
y_val = tf.keras.utils.to_categorical(valid_df.label, num_classes=2)

test_df["label"] = test_df["similarity"].apply(
        lambda x: 0 if x == 0 else 1 if x == 1 else 2
    )
y_test = tf.keras.utils.to_categorical(test_df.label, num_classes=2)



"""
## Create a custom data generator
"""


class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data.

    Args:
        sentence_pairs: Array of premise and hypothesis input sentences.
        labels: Array of labels.
        batch_size: Integer batch size.
        shuffle: boolean, whether to shuffle the data.
        include_targets: boolean, whether to incude the labels.

    Returns:
        Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
        (or just `[input_ids, attention_mask, `token_type_ids]`
         if `include_targets=False`)
    """

    def __init__(
        self,
        sentence_pairs,
        acs_pairs,
        lms_pairs,
        use_context,
        labels,
        batch_size=batch_size,
        shuffle=True,
        include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.acs_pairs = acs_pairs
        self.lms_pairs = lms_pairs
        self.use_context = use_context
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        # Load our BERT Tokenizer to encode the text.
        # We will use bert-base-uncased pretrained model.
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True
        )
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]
        acs_pairs = self.acs_pairs[indexes]
        lms_pairs = self.lms_pairs[indexes]
        use_context = self.use_context[indexes]

        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )
        # Convert batch of encoded features to numpy array.
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")
        input_acs = np.array(acs_pairs.tolist(),dtype="float32")
        input_lms = np.array(lms_pairs.tolist(),dtype="float32")
        input_use = np.array(use_context.tolist(),dtype="float32")
        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids, input_acs, input_lms, input_use], labels
        else:
            return [input_ids, attention_masks, token_type_ids, input_acs, input_lms, input_use]

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)


"""
## Build the model
"""
# Create the model under a distribution strategy scope.
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Encoded token ids from BERT tokenizer.
    input_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="input_ids"
    )
    # Attention masks indicates to the model which tokens should be attended to.
    attention_masks = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="attention_masks"
    )
    # Token type ids are binary masks identifying different sequences in the model.
    token_type_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="token_type_ids"
    )
    input_acs = tf.keras.layers.Input(
        shape=(2,), dtype=tf.float32, name="input_acs"
    )
    input_lms = tf.keras.layers.Input(
        shape=(2,), dtype=tf.float32, name="input_lms"
    )
    input_use = tf.keras.layers.Input(
        shape=(512,),  dtype=tf.float32, name="input_use"
    )
    # Loading pretrained BERT model.
    bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased", cache_dir="/tmp/cache")
    print(bert_model.config)
    # bert_model = transformers.TFBertModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
    # bert_model = transformers.TFBertModel.from_pretrained("/home/dfohr/dfohr/MMTsemantique/mmt_semantique_kerasBert/uncased_L-12_H-128_A-2")
    # Freeze the BERT model to reuse the pretrained features without modifying them.
    bert_model.trainable = False

    tmptmp = bert_model(
        input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids,
    )
    sequence_output=tmptmp.last_hidden_state

    # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
    bi_lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(sequence_output)
    # Applying hybrid pooling approach to bi_lstm sequence output.
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
    concat = tf.keras.layers.concatenate([avg_pool, max_pool])
    #dropout = tf.keras.layers.Dropout(0.3)(concat)
    #outputbert = tf.keras.layers.Dense(300, activation="relu")(dropout)
    # dropout1 = tf.keras.layers.Dropout(0.3)(input_use)
    # outputuse = tf.keras.layers.Dense(300, activation="relu")(input_use)
    concat2 =  tf.keras.layers.concatenate([concat, input_acs, input_lms, input_use])
    output = tf.keras.layers.Dense(2, activation="softmax")(concat2)
    model = tf.keras.models.Model(
        inputs=[input_ids, attention_masks, token_type_ids, input_acs, input_lms, input_use], outputs=output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["acc"],
    )


print(f"Strategy: {strategy}")
model.summary()



"""
Create train and validation data generators
"""
train_data = BertSemanticDataGenerator(
    train_df[["sentence1", "sentence2"]].values.astype("str"),
    train_df[["acs1", "acs2"]].values.astype("float"),
    train_df[["lms1", "lms2"]].values.astype("float"),
    emb_train,
    y_train,
    batch_size=batch_size,
    shuffle=True,
)


valid_data = BertSemanticDataGenerator(
    valid_df[["sentence1", "sentence2"]].values.astype("str"),
    valid_df[["acs1", "acs2"]].values.astype("float"),
    valid_df[["lms1", "lms2"]].values.astype("float"),
    emb_valid,
    y_val,
    batch_size=batch_size,
    shuffle=False,
)

## Train the Model
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=2,
    use_multiprocessing=True,
    workers=-1,
)


## Fine-tuning
# Unfreeze the bert_model.
bert_model.trainable = True

# Recompile the model to make the change effective.
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

## Train the entire model end-to-end
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)

history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=epochs,
    callbacks=[callback],
    use_multiprocessing=True,
    workers=-1,
)

# model.save_weights('save_model/model_out.h5')
outputname = outputname+"_w"
model.save_weights(outputname, save_format='tf')


exit()

## Evaluate model on the test set
test_data = BertSemanticDataGenerator(
    test_df[["sentence1", "sentence2"]].values.astype("str"),
    test_df[["acs1", "acs2"]].values.astype("float"),
    test_df[["lms1", "lms2"]].values.astype("float"),
    emb_test,
    y_test,
    batch_size=batch_size,
    shuffle=False,
)
model.evaluate(test_data, verbose=1)



## Inference on custom sentences


def check_similarity(sentence1, sentence2):
    sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
    test_data = BertSemanticDataGenerator(
        sentence_pairs, acs_pairs, lms_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
    )

    proba = model.predict(test_data)[0]
    idx = np.argmax(proba)
    proba = f"{proba[idx]: .2f}%"
    pred = labels[idx]
    return pred, proba


"""
# Check results on some example sentence pairs.
"""
sentence1 = "Two women are observing something together."
sentence2 = "Two women are standing with their eyes closed."
check_similarity(sentence1, sentence2)
"""
# Check results on some example sentence pairs.
"""
sentence1 = "A smiling costumed woman is holding an umbrella"
sentence2 = "A happy woman in a fairy costume holds an umbrella"
check_similarity(sentence1, sentence2)

"""
#Check results on some example sentence pairs
"""
sentence1 = "A soccer game with multiple males playing"
sentence2 = "Some men are playing a sport"
check_similarity(sentence1, sentence2)