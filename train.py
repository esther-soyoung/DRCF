from __future__ import unicode_literals, print_function, division
import numpy as np
import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn as nn
import model
import utils
import sys
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

torch.manual_seed(1)
np.random.seed(1)

# ==============================================
## DATA PATH
PATH = "./data/"
DATA = 'taxi'  # ny, la, taxi
TRAIN_DATA_PATH = PATH + DATA + '/drcf_' + DATA + '_train.tsv'
VALIDATAION_DATA_PATH = PATH + DATA + '/drcf_' + DATA + '_valid.tsv'
TEST_DATA_PATH = PATH + DATA + '/drcf_' + DATA + '_test.tsv'

## HYPER-PARAMETERS
EMBEDDING_DIM = 50
RNN_STEP = 5
EPOCHS = 1000
BATCH_SIZE = 2000
LEARNING_RATE = 0.001
SAMPLE_NUM = 8

# =============================================
## DATA PREPARATION
print("========================================")
print("Data Loading..")
train, validation, test = utils.load_data(TRAIN_DATA_PATH, VALIDATAION_DATA_PATH, TEST_DATA_PATH)

print("Make Dictionary..")
train, validation, test, user2id, id2user, venue2id, id2venue, venue_frequency = utils.make_dict(train, validation, test)

print("Make Input..")
train, validation, test = utils.make_input(train, validation, test, RNN_STEP)

# =============================================
def get_eval_score(candidate, rank):
	_mrr = .0

	for i in xrange(len(candidate)):
		_rank = np.where(rank[i] == candidate[i])
		_mrr += (1.0/(_rank[0]+1))

	return _mrr

def get_acc(candidate, rank):
    """target and scores are torch cuda Variable"""
    acc = np.zeros((4, 1))
    for i, p in enumerate(rank):  # enumerate for the number of targets
		acc[3] += 1
		t = candidate[i]
		if t in p[:10] and t > 0:
			acc[2] += 1  # top10
		if t in p[:5] and t > 0:
			acc[1] += 1  # top5
		if t == p[0] and t > 0:
			acc[0] += 1  # top1
    return acc

## Training
print("========================================")
print("Training..")

drcf = model.DRCF(EMBEDDING_DIM, RNN_STEP, len(user2id), len(venue2id), SAMPLE_NUM)
drcf = nn.DataParallel(drcf).cuda()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, drcf.parameters()), lr=LEARNING_RATE)
criterion = nn.LogSigmoid()

for i in xrange(EPOCHS):
	# Training
	drcf.train()
	step = 0
	loss = .0
	batch_num = int(len(train)/BATCH_SIZE) + 1

	batches = utils.batches(train, BATCH_SIZE, SAMPLE_NUM, venue_frequency)
	for batch in batches:
		user, candidate, checkins, samples = batch
		input_user = Variable(torch.cuda.LongTensor(user))
		input_candidate = Variable(torch.cuda.LongTensor(candidate))
		input_checkins = Variable(torch.cuda.LongTensor(checkins))
		input_samples = Variable(torch.cuda.LongTensor(samples))

		# Optimizing
		optimizer.zero_grad()
		_loss = -criterion(drcf(input_user, input_candidate, input_checkins, input_samples)).sum()
		_loss.backward()
		optimizer.step()
		# loss+=_loss.cpu().data.numpy()[0]
		loss+=_loss.cpu().data.numpy()

		# Printing Progress
		step+=1
		sys.stdout.write("\033[F")
		sys.stdout.write("\033[K")
		print("Process Training Epoch: [{}/{}] Batch: [{}/{}] Loss: {}".format(i+1, EPOCHS, step, batch_num, _loss.cpu().data.numpy()))


	if (i+1) % 10 == 0:
		with torch.no_grad():
			# Validation
			drcf.eval()
			step = 0
			mrr = .0
			batch_num = int(len(validation)/100) + 1

			batches = utils.batches(validation, 100, SAMPLE_NUM, venue_frequency)
			acc = [0, 0, 0, 0]  # top1, top5, top10, tot
			for batch in batches:
				user, candidate, checkins, _ = batch
				input_user = Variable(torch.cuda.LongTensor(user))
				input_checkins = Variable(torch.cuda.LongTensor(checkins))
				
				# Optimizing
				out, rank = drcf.module.evaluation(input_user, input_checkins)
				rank = rank.cpu().data.numpy()
				out = out.cpu().numpy()  # (100, dim)
				mrr += get_eval_score(candidate, rank)
				batch_acc = get_acc(candidate, rank)
				acc[0] += batch_acc[0]
				acc[1] += batch_acc[1]
				acc[2] += batch_acc[2]
				acc[3] += batch_acc[3]

				# Printing Progress
				step+=1
				sys.stdout.write("\033[F")
				sys.stdout.write("\033[K")
				print("Process Evaluation Epoch: [{}/{}] Batch: [{}/{}] Batch Top1Acc: [{}]\n".format(i+1, EPOCHS, step, batch_num, batch_acc[0]/batch_acc[3]))

			sys.stdout.write("\033[F")
			sys.stdout.write("\033[K")
			print("Process Epoch: [{}/{}] loss : [{}] Top1Acc: [{}] Top5Acc: [{}] Top10Acc: [{}]\n".format(i+1, EPOCHS, loss, acc[0]/acc[3], acc[1]/acc[3], acc[2]/acc[3]))

	else:
		sys.stdout.write("\033[F")
		sys.stdout.write("\033[K")
		print("Process Epoch: [{}/{}] loss : [{}]\n".format(i+1, EPOCHS, loss))

print("========================================")
print("Saving..")
torch.save(drcf.state_dict(), 'model_' + DATA + '.m')

print("========================================")
print("Testing..")
with torch.no_grad():
	drcf.eval()
	step = 0
	mrr = .0
	batch_num = int(len(validation)/100) + 1

	batches = utils.batches(validation, 100, SAMPLE_NUM, venue_frequency)
	acc = [0, 0, 0, 0]  # top1, top5, top10, tot
	ndcg = [0, 0, 0]  # @1, @5, @10
	for batch in batches:
		user, candidate, checkins, _ = batch
		input_user = Variable(torch.cuda.LongTensor(user))
		input_checkins = Variable(torch.cuda.LongTensor(checkins))
		
		# Optimizing
		_, rank = drcf.module.evaluation(input_user, input_checkins)
		rank = rank.cpu().data.numpy()
		r = _.cpu().numpy()  # (100, dim)
		mrr += get_eval_score(candidate, rank)
		batch_acc = get_acc(candidate, rank)
		acc[0] += batch_acc[0]
		acc[1] += batch_acc[1]
		acc[2] += batch_acc[2]
		acc[3] += batch_acc[3]

		for i in r:
			ndcg[0] += utils.ndcg_at_k(i, 1)
			ndcg[1] += utils.ndcg_at_k(i, 5)
			ndcg[2] += utils.ndcg_at_k(i, 10)

		# Printing Progress
		step+=1
		sys.stdout.write("\033[F")
		sys.stdout.write("\033[K")
		print("Batch: [{}/{}] Batch Top1Acc: [{}]\n".format(step, batch_num, batch_acc[0]/batch_acc[3]))

	sys.stdout.write("\033[F")
	sys.stdout.write("\033[K")
	print("Loss : [{}] Top1Acc: [{}] Top5Acc: [{}] Top10Acc: [{}]".format(loss, acc[0]/acc[3], acc[1]/acc[3], acc[2]/acc[3]))
	print("NDCG@1: [{}] NDCG@5: [{}] NDCG@10: [{}]".format(ndcg[0]/acc[3][0], ndcg[1]/acc[3][0], ndcg[2]/acc[3][0]))