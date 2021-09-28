"""
Title: Semantic Similarity with BERT
Author: [Mohamad Merchant](https://twitter.com/mohmadmerchant1)
Date created: 2020/08/15
Last modified: 2020/08/29
Description: Natural Language Inference by fine-tuning BERT model on SNLI Corpus.
"""
"""
## Introduction

Semantic Similarity is the task of determining how similar
two sentences are, in terms of what they mean.
This example demonstrates the use of SNLI (Stanford Natural Language Inference) Corpus
to predict sentence semantic similarity with Transformers.
We will fine-tune a BERT model that takes two sentences as inputs
and that outputs a similarity score for these two sentences.

### References

* [BERT](https://arxiv.org/pdf/1810.04805.pdf)
* [SNLI](https://nlp.stanford.edu/projects/snli/)
"""

"""
## Setup

Note: install HuggingFace `transformers` via `pip install transformers` (version >= 2.11.0).
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
import h5py
import getopt, sys

def usage():
    print("usage --model= --hdf5= --out=res")
    print("       test de reco avec DNN ")

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hdiloms:v", ["help", "model=", "hdf5=", "out="])
    except getopt.GetoptError as err:
        # hdf5 is contain hypothesis which we want to compare
        # print help information and exit
        # "out"- it contains results that we want to obtain
        # "model" - Here we used DNN model that do the prediction
        print(str(err))  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    nomfichmodel = None
    nomfichhdf5 = None
    nomfichoutput = None
    for o, a in opts:
        if o == "-v":
            verbose = True
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("--hdf5"):
            nomfichhdf5 = a
        elif o in ("-o", "--out"):
            nomfichoutput = a
        elif o in ("--model"):
            nomfichmodel = a
    if(nomfichhdf5 == None):
        print('--hdf5 missing')
        usage()
        sys.exit(2)
    if(nomfichoutput == None):
        print('--out missing')
        usage()
        sys.exit(2)
    if(nomfichmodel == None):
        print('--model missing')
        usage()
        sys.exit(2)

    """
    ## Configuration
    $"""

    max_length = 128  # Maximum length of input sentence to the model.
    batch_size = 1

    # Labels in our dataset.
    labels = ["0", "1" ]

    """
    ## Load the Data
    """

    """
    ## Preprocessing
    """

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
            labels,
            batch_size=batch_size,
            shuffle=True,
            include_targets=False,
        ):
            self.sentence_pairs = sentence_pairs
            self.acs_pairs = acs_pairs
            self.lms_pairs = lms_pairs
            self.labels = labels
            self.shuffle = shuffle
            self.batch_size = batch_size
            self.include_targets = include_targets
            # Load our BERT Tokenizer to encode the text.
            # We will use base-base-uncased pretrained model.
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
            # With BERT tokenizer's batch_encode_plus batch of both the sentences are
            # encoded together and separated by [SEP] token.
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
            # Set to true if data generator is used for training/validation.
            if self.include_targets:
                labels = np.array(self.labels[indexes], dtype="int32")
                return [input_ids, attention_masks, token_type_ids, input_acs, input_lms], labels
            else:
                return [input_ids, attention_masks, token_type_ids, input_acs, input_lms]

        def on_epoch_end(self):
            # Shuffle indexes after each epoch if shuffle is set to True.
            if self.shuffle:
                np.random.RandomState(42).shuffle(self.indexes)


    """
    ## Build the model
    """
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
    # Loading pretrained BERT model.
    # Here we used Bert-base-uncased model.
    bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")
    # Freeze the BERT model to reuse the pretrained features without modifying them.
    bert_model.trainable = True

    tmpmod = bert_model(
        input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
    )
    sequence_output = tmpmod.last_hidden_state
    # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
    # We add Bi-Lstm model on the top of the bert model, we set hidden size as 64 and it take input sequence from bert model i.e output of the BERT model.
    bi_lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(sequence_output)
    # Applying hybrid pooling approach to bi_lstm sequence output.
    # It take the output from the bi-lstm layer and compute average pooling i.e in the form of vector.
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
    # Also we used MaxPooling function to obtain max which is dimention by dimention.
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
    # We concatenate each vector produced by average pooling and max pooling to get an single vector.
    concat = tf.keras.layers.concatenate([avg_pool, max_pool])
    # We set dropout as 0.3, randomly it takes 30 % of the input vector as 0.
    dropout = tf.keras.layers.Dropout(0.3)(concat)
    # Here we optain the output of the model.
    # We added fully connected layer for the input from the dropout layer, and set size of the output is 300.
    outputbert = tf.keras.layers.Dense(300, activation="softmax")(dropout)
    # Finaly we cancatenate output vector of the model that we obtain, acoustic vector and linguistic vector.
    concat2 =  tf.keras.layers.concatenate([outputbert, input_acs, input_lms])
    # Finally we add output layer for the final model.
    output = tf.keras.layers.Dense(2, activation="softmax")(concat2)
    model = tf.keras.models.Model(
        inputs=[input_ids, attention_masks, token_type_ids, input_acs, input_lms], outputs=output
    )
    # Here we compile the model.

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["acc"],
    )

    model.summary()
    # Here we load the weights of the model that we obtain.
    model.load_weights(nomfichmodel)
    # Read the data
    fichdevtest = h5py.File(nomfichhdf5, "r")

    nbtotpairs = 0
    nbmaxpair = 0
    nbmaxhyp = 0
    # lecture du fichier des frames
    # Here we iterate on the 8 conference files.
    for loc in fichdevtest.keys():
        # Here we iterate sentence for each conferences.
        for sent in fichdevtest[loc]:
            nb = fichdevtest[loc][sent]['ac_score'].shape[0]
            print(nb)
            if(nb>nbmaxhyp):
                nbmaxhyp = nb
            nbpairs = nb*(nb-1)/2
            nbtotpairs += nbpairs
            if(nbpairs> nbmaxpair):
                nbmaxpair = int(nbpairs)

    print("nbtotpairs",nbtotpairs," nbmaxpair",nbmaxpair," nbmaxhyp",nbmaxhyp)
    ll = list(fichdevtest)
    print("nb loc",ll)

    nbhyp       =  np.zeros((nbmaxpair, 2), dtype=np.int64)
    accscores   =  np.zeros((nbmaxpair, 2), dtype=np.float32)
    lmscores    =  np.zeros((nbmaxpair, 2), dtype=np.float32)
    norm_acscores =  np.zeros((nbmaxhyp, ), dtype=np.float32)
    norm_lmscores =  np.zeros((nbmaxhyp, ), dtype=np.float32)
    # cls       = np.zeros((nbmaxpair, taille_npairs), dtype="float32")

    # y_devtest =  np.zeros((nbtotpairs,), dtype=np.float32)
    wer       =  np.zeros((nbmaxpair, 2), dtype=np.float32)
    nb1 = 0
    score_bert = np.zeros((nb,), dtype=np.float32)

    fichres = open(nomfichoutput, "w")
    fichres.write("loc,sentence,hypothese,score\n")
    for loc in fichdevtest.keys():
        for sent in fichdevtest[loc]:
            nb = fichdevtest[loc][sent]['ac_score'].shape[0]
            nbpairs = nb*(nb-1)/2
            sent_id = []
            texthyp = []
            y_label = []
            ipaire = 0

            # normalisation des scores
            # Here we did the normalization for 20 hypothesis.
            accmax = -1e20
            for s in range(nb):
                norm_acscores[s]= 0 - fichdevtest[loc][sent]['ac_score'][s]
                if norm_acscores[s]>accmax:
                    accmax=norm_acscores[s]
            for s in range(nb):
                #  After obtain normalization score, we remove the normalization score from the acoustic score.
                norm_acscores[s]= norm_acscores[s] - accmax
            # same we did normalization for linguistic score
            lmmax = -1e20
            for s in range(nb):
                norm_lmscores[s]= 0 - fichdevtest[loc][sent]['lm_score'][s]
                if norm_lmscores[s]>lmmax:
                    lmmax=norm_lmscores[s]
            for s in range(nb):
                norm_lmscores[s]= norm_lmscores[s] - lmmax
            # Here we construct all the pair of hypothesis.
            for hyp1 in range(nb):
                for hyp2 in range (nb):
                    if(hyp1>hyp2):
                        # print("hyp1",hyp1,"hyp2",hyp2)
                        # text2hyp is list of two hypothesis.
                        text2hyp = []
                        text2hyp.append(fichdevtest[loc][sent]['texthyp'][hyp1])
                        text2hyp.append(fichdevtest[loc][sent]['texthyp'][hyp2])
                        texthyp.append(text2hyp)
                        # Here we obtain acoustic score for two hypothesis.
                        accscores[ipaire][0]=norm_acscores[hyp1]
                        accscores[ipaire][1]=norm_acscores[hyp2]
                        # We obetain linguistic score for two hypothesis.
                        lmscores[ipaire][0]=norm_lmscores[hyp1]
                        lmscores[ipaire][1]=norm_lmscores[hyp2]
                        nbhyp[ipaire][0]=hyp1
                        nbhyp[ipaire][1]=hyp2
                        ipaire += 1


            print("nb*(nb-1)/2", nbpairs, "len(texthyp)", len(texthyp))
            test_array_hyp_pairs = np.array(texthyp)
            print("accscores.shape",accscores.shape)
            """
            ## Evaluate model on the test set
            """
            test_data = BertSemanticDataGenerator(
                test_array_hyp_pairs,
                accscores,
                lmscores,
                y_label,
                batch_size=batch_size,
                shuffle=False,
                include_targets=False
            )
            # print("test_data[0]",test_data[0])
            # Here we do the prediction for the test data (for all the pairs)
            predictions = model.predict(test_data)
            print("predictions.shape",predictions.shape)
            if(predictions.shape[0] != ipaire):
                print("pb prediction", predictions.shape,ipaire)
                isent=0
                sent_courante = listpairsent_id[0]
                sys.exit(2)

            for j in range(nb):
                score_bert[j] = 0.0

            texthypothese = []
            for i in range(nb):
                texthypothese.append("UNK")

            # cumul des scores pour chaque hypothese
            # cumulative scores for each hypothesis
            for i in range(ipaire):
                h1 = nbhyp[i][0]
                # list_hyp.add(h1)
                texthypothese[h1] = test_array_hyp_pairs[i][0]
                h2 = nbhyp[i][1]
                # list_hyp.add(h2)
                texthypothese[h2] = test_array_hyp_pairs[i][1]
                # DNN predit 1 si deuxieme meilleure que premiere
                score_bert[h1] += predictions[i][0]
                score_bert[h2] += predictions[i][1]
                # lscore_ac[h1] = ac_score[i][0]
                # lscore_ac[h2] = ac_score[i][1]
                # lscore_lm[h1] = lm_score[i][0]
                # lscore_lm[h2] = lm_score[i][1]
                # print("h1",h1,"h2",h2,score_bert[h1],score_bert[h2])
                # print(score)

            hypgagnante = -1
            mx=-1e20
            # Finally we obtain bert score.
            print(score_bert)
            for h in range(nb):
                fichres.write(loc)
                fichres.write(",")
                fichres.write(sent)
                fichres.write(",")
                fichres.write(str(h))
                fichres.write(",")
                fichres.write(str(score_bert[h]/(nb-1.)))
                fichres.write("\n")
    fichres.close()

if __name__ == "__main__":
    main()