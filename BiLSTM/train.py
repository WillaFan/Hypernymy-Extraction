from preprocess import DataReader, gen_embedding_from_file, read_tag_vocab
from config import config, apply_random_seed
from model import sequence_labeling
from tqdm import tqdm
import torch
from evalu import evaluate
import matplotlib.pyplot as plt


if __name__ == "__main__": 
	_config = config()
	apply_random_seed()

	tag_dict = read_tag_vocab(_config.tag_file)
	reversed_tag_dict = {v: k for (k, v) in tag_dict.items()}
	word_embedding, word_dict = gen_embedding_from_file(_config.word_embedding_file, _config.word_embedding_dim)
	char_embedding, char_dict = gen_embedding_from_file(_config.char_embedding_file, _config.char_embedding_dim)

	_config.nwords = len(word_dict)
	_config.ntags = len(tag_dict)
	_config.nchars = len(char_dict)

	# read training and development data
	train = DataReader(_config, _config.train_file, word_dict, char_dict, tag_dict, _config.batch_size, is_train=True)
	dev = DataReader(_config, _config.dev_file, word_dict, char_dict, tag_dict, _config.batch_size)

	# compare the optimizers
	mod = sequence_labeling(_config, word_embedding, char_embedding)
	optims={}
	optims['SGD'] = torch.optim.SGD(mod.parameters(), lr=0.1)
	optims['Adam'] = torch.optim.Adam(mod.parameters())
	optims['RMSprop'] = torch.optim.RMSprop(mod.parameters())
	# optimizer = torch.optim.Adam(model.parameters())

	best_f1 = 0.0
	train_loss={}
	for key in optims.keys():
		train_loss[key]=[]
		for i in range(_config.nepoch):
			mod.train()
			print('Epoch %d / %d' % (i + 1, _config.nepoch))
			# you can disable pbar if you do not want to show the training progress
			with tqdm(total=len(train)) as pbar:
				for batch_sentence_len_list, batch_word_index_lists, batch_word_mask, batch_char_index_matrices, batch_char_mask, batch_word_len_lists, batch_tag_index_list in train:
					optims[key].zero_grad()
					loss = mod(batch_word_index_lists, batch_sentence_len_list, batch_word_mask, batch_char_index_matrices, batch_word_len_lists, batch_char_mask, batch_tag_index_list)
					train_loss[key].append(loss.item())
					loss.backward()
					optims[key].step()
					pbar.set_description('loss %.4f' % loss.view(-1).data.tolist()[0])
					pbar.update(1)
	colors={"SGD": 'green', "Adam": 'blue', "RMSprop": 'pink'}
	x = range(len(train)*10)
	for key in optims.keys():
		plt.plot(x, train_loss[key], color=colors[key], label=key)
	plt.xlabel("iterations")
	plt.ylabel("loss")
	plt.legend()
	plt.title("Bi-LSTM Hyponymy Classification Training Loss")
	plt.grid(True)
	plt.savefig("./loss.png")
	plt.close()

	model = sequence_labeling(_config, word_embedding, char_embedding)
	optimizer = torch.optim.Adam(model.parameters())
	f1_train = []; f1_dev = []
	for i in range(_config.nepoch):
		model.train()
		print('Epoch %d / %d' % (i + 1, _config.nepoch))
		pred_train_ins, golden_train_ins = [], []
		with tqdm(total=len(train)) as pbar:
			for batch_sentence_len_list, batch_word_index_lists, batch_word_mask, batch_char_index_matrices, batch_char_mask, batch_word_len_lists, batch_tag_index_list in train:
				optimizer.zero_grad()
				loss = model(batch_word_index_lists, batch_sentence_len_list, batch_word_mask, batch_char_index_matrices, batch_word_len_lists, batch_char_mask, batch_tag_index_list)
				loss.backward()
				optimizer.step()
				pbar.set_description('loss %.4f' % loss.view(-1).data.tolist()[0])
				pbar.update(1)
				pred_batch_tag = model.decode(batch_word_index_lists, batch_sentence_len_list,
											  batch_char_index_matrices, batch_word_len_lists, batch_char_mask)
				pred_train_ins += [[reversed_tag_dict[t] for t in tag[:l]] for tag, l in
								 zip(pred_batch_tag.data.tolist(), batch_sentence_len_list.data.tolist())]
				golden_train_ins += [[reversed_tag_dict[t] for t in tag[:l]] for tag, l in
								   zip(batch_tag_index_list.data.tolist(), batch_sentence_len_list.data.tolist())]
			new_f1_train = evaluate(golden_train_ins, pred_train_ins)
			f1_train.append(new_f1_train)

		# keep the model with best f1 on development set, if the flag is True
		if _config.use_f1:
			model.eval()
			pred_dev_ins, golden_dev_ins = [], []
			for batch_sentence_len_list, batch_word_index_lists, batch_word_mask, batch_char_index_matrices, batch_char_mask, batch_word_len_lists, batch_tag_index_list in dev:
				pred_batch_tag = model.decode(batch_word_index_lists, batch_sentence_len_list, batch_char_index_matrices, batch_word_len_lists, batch_char_mask)
				pred_dev_ins += [[reversed_tag_dict[t] for t in tag[:l]] for tag, l in zip(pred_batch_tag.data.tolist(), batch_sentence_len_list.data.tolist())]
				golden_dev_ins += [[reversed_tag_dict[t] for t in tag[:l]] for tag, l in zip(batch_tag_index_list.data.tolist(), batch_sentence_len_list.data.tolist())]

			new_f1 = evaluate(golden_dev_ins, pred_dev_ins)
			f1_dev.append(new_f1)
			if new_f1 > best_f1:
				model_state = model.state_dict()
				# torch.save(model_state, _config.model_file)
				torch.save(model, _config.model_file)
				best_f1 = new_f1
		# else we just keep the newest model
		else:
			model_state = model.state_dict()
			torch.save(model, _config.model_file)

	col = {'Training set':'blue', 'Validation set':'green'}
	for f1, name in zip([f1_train,f1_dev],['Training set','Validation set']):
		plt.plot(range(_config.nepoch), f1, color=col[name], label=name)
	plt.title('F1 score in Training')
	plt.xlabel('epoch')
	plt.ylabel('f1 score')
	plt.legend()
	plt.grid(True)
	plt.savefig("./f1.png")