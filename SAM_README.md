# 244 Assignment 1 Readme


## Getting Setup:

0. Make sure that conda/miniconda is installed on your machine.
1. Make sure that cmake and Rust are installed on your machine.
2. Run the setup script: `./setup.sh`


## Running the Code:

...



## Sam Notes

- We're to implement code in `llama.py`, `classifier.py`, and `optimizer.py`.
    - I believe we also have to implement RoPE in `rope.py`, but this isn't in the assignment instructions.
- In terms of order, it makes intuitive sense to implement:
    - `llama.py`
    - `rope.py`
    - `classifier.py`
    - `optimizer.py`
- (The last two might be swapped, I'm not sure yet)
- We're starting from a pretrained model in `stores42.pt`, which is a 42M parameter model pretraiend on TinyStories (the dataset from the Phi 1 model, iirc). 
- Once we've implemented these components, test our models in 3 settings:
    - Generate text completions.
    - Performing zero-shot prompt-based sentiment analysis on two datasets (will give bad results).
    - Perform task-specific finetuning after implementing a classification head in `classifier.py`.
- Optionally, we can try implementing something new on top of this language model!