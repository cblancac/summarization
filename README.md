# summarization

## :gear: Setup
- Clone the repository: `https://github.com/cblancac/summarization.git`.
- `pip install -r requirements.txt`
- `pip install --no-cache-dir transformers sentencepiece`


## 📚 Dataset
The dataset used in this project is the **SAMSum dataset** developed by Samsung, which consists of a collection of dialogues along with brief summaries. In an enterprise setting, these dialogues might represent the interactions between a customer and the support center, so generating accurate summaries can help improve customer service and detect common patterns among customer requests. Let's look an example from this dataset:
![dataset](https://github.com/cblancac/SentimentAnalysisBert/assets/105242658/51f63da7-e0e9-4aff-bc02-07854187116d)

The dialogues look like what you would expect from a chat via SMS or WhatsApp, including emojis and placeholders for GIFs. The dialogue field contains the full text and the summary the summarized dialogue.


## :brain: Summarization
Text summarization is a difficult task for neural language models, including transformers. Despite these challenges, text summarization offers the prospect for domain experts to significantly speed up their workflows and is used by enterprises to condense internal knowledge, summarize contracts, automatically generate content for social media releases, and more. In this chapter, it is explored explore how pretrained transformers can be leveraged to summarize documents. Summarization is a classic sequence-to-sequence (seq2seq) task with an input text and a target text (this is where encoder-decoder transformers excel). Our own encoder-decoder model will be created to condense dialogues
between several people into a crisp summary.

The encoder-decoder transformer choosen for this project is **PEGASUS**. As shown in the picture below, its pretraining objective is to predict masked sentences in multisentence texts. The closer the pretraining objective is to the downstream task, the more effective it is. With the aim of finding a pretraining objective that is closer to summarization than general language modeling, they automatically identified, in a very large corpus, sentences containing most of the content of their surrounding paragraphs (using summarization evaluation metrics as a heuristic for content overlap) and pretrained the PEGASUS model to reconstruct these sentences, thereby obtaining a state-of-the-art model for text summarization.

![pegasus](https://github.com/cblancac/SentimentAnalysisBert/assets/105242658/cf539397-0ecb-458d-8eea-d22ead46c12b)


## :weight_lifting_man: Training models
The model choosen as base model is "google/pegasus-cnn_dailymail", which can be found in Hugging Face. This model was fine-tuned with the CNN/DailyMail dataset, which consists of around 300,000 pairs of news articles and
their corresponding summaries, composed from the bullet points that CNN and the DailyMail attach to their articles. An important aspect of the dataset is that the summaries are abstractive and not extractive, which means that they consist of new sentences instead of simple excerpts.

This model is fine-tuned in this project to create good summarization for dialogues

