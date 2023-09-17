# summarization

## :gear: Setup
- Clone the repository: `https://github.com/cblancac/summarization.git`.
- `pip install -r requirements.txt`
- `pip install --no-cache-dir transformers sentencepiece`


## :📚: Dataset

## :brain: Summarization
Text summarization is a difficult task for neural language models, including transformers. Despite these challenges, text summarization offers the prospect for domain experts to significantly speed up their workflows and is used by enterprises to condense internal knowledge, summarize contracts, automatically generate content for social media releases, and more. In this chapter, it is explored explore how pretrained transformers can be leveraged to summarize documents. Summarization is a classic sequence-to-sequence (seq2seq) task with an input text and a target text (this is where encoder-decoder transformers excel). Our own encoder-decoder model will be created to condense dialogues
between several people into a crisp summary.

The encoder-decoder transformer choosen for this project is **PEGASUS**. As shown in the picture below, its pretraining objective is to predict masked sentences in multisentence texts. The closer the pretraining objective is to the downstream task, the more effective it is. With the aim of finding a pretraining objective that is closer to summarization than general language modeling, they automatically identified, in a very large corpus, sentences containing most of the content of their surrounding paragraphs (using summarization evaluation metrics as a heuristic for content overlap) and pretrained the PEGASUS model to reconstruct these sentences, thereby
obtaining a state-of-the-art model for text summarization.

![pegasus](https://github.com/cblancac/SentimentAnalysisBert/assets/105242658/cf539397-0ecb-458d-8eea-d22ead46c12b)

