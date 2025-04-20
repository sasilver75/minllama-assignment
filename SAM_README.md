# 244 Assignment 1 Readme


## Getting Setup:

0. Make sure that conda/miniconda is installed on your machine.
1. Make sure that cmake and Rust are installed on your machine.
2. Run the setup script: `./setup.sh`


## File Structure
Some of the general files are:
- `config.py`: Defines a LlamaConfig class holding all hyperparameters.
- `base_llama.py`: Provides a lightweight "pretrained model" base class and wiring for saving and loading configs.
- `utils.py`: Miscellaneous helpers

The four files that we really need to worry about are:
- `rope.py`: Implements Rotary Positional Embeddings
- `llama.py`: The heart of the assignment. A from-scratch PyTorch implmention of MH, GQA transformer blocks and the full LLama 2 mode.
    - RMSNorm
    - Attention
    - FeedForward
    - LlamaLayer (Layer Norm > Attention > Residual > Layer Norm > FFN > Residual)
    - Llama (token embedding, stack of LlamaLayers, final norm + output head, weight tying, and a generate() function stub)
    - load_pretrained() to load a checkpoint into your model.
    - Bits where we have to implement are usually marked with `raise Implemented`
- `classifier.py`: Two PyTorch Modules that wrap a frozen or trainable Llama backbone for classification: LlamaZeroShotClassifier and LlamaEmbedingClassifier.
- `optimizer`: A skeleton for our own AdamW optimizer, mimicking PyTorch's but implementing it from scratch.


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