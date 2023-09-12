from transformers import pipeline


pipe = pipeline("summarization", model="pegasus-finetuned")
gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}

CUSTOM_DIALOGUE = """
Hannah: Hey, do you have Betty's number?
Amanda: Lemme check
Hannah: <file_gif>
Amanda: Sorry, can't find it.
Amanda: Ask Larry
Amanda: He called her last time we were at the park together
Hannah: I don't know him well
Hannah: <file_gif>
Amanda: Don't be shy, he's very nice
Hannah: If you say so..
Hannah: I'd rather you texted him
Amanda: Just text him
Hannah: Urgh.. Alright
Hannah: Bye
Amanda: Bye bye
"""


if __name__ == "__main__":
    print(pipe(CUSTOM_DIALOGUE, **gen_kwargs)[0]["summary_text"])
