# Automatic-Speech-Recognition

This research intends to improve the Automated Speech Recognition System (ASR). Due to lack of noise from various sound environments like microphone change and sound capture equipment etc., the Word Error Rate is increased, and so the capacity of a machine to detect speech is significantly inferior to that of a
human. The acoustic information may be less reliable in this situation.

After the many experiments conducted on Automatic Speech Recognition Systems by various authors, The ASR system mainly focuses on Acoustic, Linguistic, and Semantic models. To produce an accurate prediction of transcription,
we would propose a new approach to the semantic model in an ASR system.

This approach re-evaluate the N-best hypothesis list by adding more semantic information to the ASR system, in such a way that adding the previous sentence to recognize the current sentence. Sometimes, the previous sentence might be helpful
for understanding and recognizing the current sentence. And in this experiment we would check semantic information from the past in a Speech Recognition System: does the past help the present?

To achieve this, we performed re-scoring the ASR N-best hypothesis list with the help of Deep Neural Network (DNN) models and combined with acoustic,and linguistic information. The goal of our DNN res-coring models is to find
hypotheses with improved semantic consistency and thus lower Word Error Rate. We used the publicly accessible TED-LIUM corpus to predict semantic scores in the hopes of enhancing the ASR system. In this approach, we employed a transformer extension called BERT, and Universal Sentence Encoder for continuous
representation of sentence embedding plays a major role.

The goal of our DNN re-scoring models is to find hypotheses with improved semantic consistency and thus lower Word Error Rate. We evaluated our methodology on the corpus of TED-LIUM conferences with noise of SNR 10 dB. The proposed model
gives a significant WER by using past information as compared to without using  past information. The best Word Error Rate of BERTpast re-scoring models is 13. As compared to previous work in this same task, our proposed model improvements are statistically significant.
