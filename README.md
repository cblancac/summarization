# summarization

## :gear: Setup
- Clone the repository: `https://github.com/cblancac/summarization.git`.
- `pip install -r requirements.txt`
- `pip install --no-cache-dir transformers sentencepiece`


## :brain: Summarization
Text summarization is a difficult task for neural language models, including transformers. Despite these challenges, text summarization offers the prospect for domain experts to significantly speed up their workflows and is used by enterprises to condense internal knowledge, summarize contracts, automatically generate content for social media releases, and more. In this chapter, it is explored explore how pretrained transformers can be leveraged to summarize documents. Summarization is a classic sequence-to-sequence (seq2seq) task with an input text and a target text (this is where encoder-decoder transformers excel). Our own encoder-decoder model will be created to condense dialogues
between several people into a crisp summary.

![pegasus](https://github.com/cblancac/SentimentAnalysisBert/assets/105242658/c004185c-eb8b-4fde-9732-343cde1e7e6e)
