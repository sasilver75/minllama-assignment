
import torch
import torch.nn.functional as F

# change it with respect to the original model
from config import LlamaConfig
from llama import load_pretrained
from tokenizer import Tokenizer

class LlamaZeroShotClassifier(torch.nn.Module):
	"""
	Init: Loads a frozen Llama model and encodes each human-readable label into a sequence of token ids
	Forward: Given a batch (eg bs=1) of sequences of token ids...
		There are N possible labels, e.g. "Cool Guy", "Total Loser", each of which might be multiple tokens.
		We compute the probability of any of these subsequences following the given sequence (a prompt)
		We simply sub the log probabilities of each {prompt sequence}+{label_sequence}
	"""
	def __init__(self, config: LlamaConfig, tokenizer: Tokenizer, label_names: list[str]):
		super(LlamaZeroShotClassifier, self).__init__()
		self.num_labels = config.num_labels
		self.llama = load_pretrained(config.pretrained_model_path)
		# Zero-shot classification does not require updating llama paramters.
		for param in self.llama.parameters():
			param.requires_grad = False
		assert len(label_names) == self.num_labels
		self.tokenizer = tokenizer
		self.label_name_ids = [tokenizer.encode(label, bos=False, eos=False) for label in label_names]


	def forward(self, input_ids):
		# compute the completion probability of each label string
		logits, _ = self.llama(input_ids)
		log_probabilities = F.log_softmax(logits, dim=-1)
		label_probabilities = torch.zeros((log_probabilities.shape[0], self.num_labels), device=log_probabilities.device)
		for i, label_token_ids in enumerate(self.label_name_ids):
			total_log_prob = torch.sum(log_probabilities[:, :, label_token_ids], axis=-1)
			label_probabilities[:, i] = total_log_prob[:, 0]
		return label_probabilities

class LlamaEmbeddingClassifier(torch.nn.Module):
	def __init__(self, config):
		super(LlamaEmbeddingClassifier, self).__init__()
		self.num_labels = config.num_labels
		self.llama = load_pretrained(config.pretrained_model_path)
		# If we use pretrain mode, we freeze Llama parameters.
		for param in self.llama.parameters():
			if config.option == 'pretrain':
				param.requires_grad = False
			elif config.option == 'finetune':
				param.requires_grad = True

		self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
		self.classifier_head = torch.nn.Linear(self.llama.config.dim, self.num_labels)

	def forward(self, input_ids):
		'''
		1) Find the hidden state after the final token of the input sequence
		2) Apply dropout (self.dropout) to the hidden state at training time to mitigate
		   overfitting.
		2) Pass this through the classifier head (self.classifier_head), which will return
		   logits (unnormalized probabilities) over all classes.
		3) Take the log-softmax of the logits and return log-probabilities over all classes.
		'''
		"""
		Again, input_ids is a tensor of shape (bs, batch_max_sequence_length), where each entry
		is an integer token ID from the model's vocabulary.
		"""
		# logits is (bs, current_seq_len, vocab), unless in self.forward targets is None (which it usually is), then it's (bs, 1, vocab)
		_, h = self.llama(input_ids) # h is a (bs, seq_len, hdim) tensor
		
		# take just the final token's hidden state; we just care about the last token
		h_last_token = h[:, -1, :] # (bs, hdim)
		
		# Dropout
		h_last_token_dropout = self.dropout(h_last_token) # apply dropout as instructed; (bs, hdim)

		# Project to vocab dim
		logits = self.classifier_head(h_last_token_dropout) # (bs, vocab) classifier head is just a linear layer from hdim to vocab
		
		# Get log probs and return
		log_probs = F.log_softmax(logits, dim=-1) # (bs, vocab)
		return log_probs