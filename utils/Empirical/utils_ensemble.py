import shutil
import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.autograd as autograd
import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.append(parentdir)
from tqdm import tqdm
import PIL.Image
from torchvision.transforms import ToTensor
import itertools
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from advertorch.attacks import GradientSignAttack, LinfBasicIterativeAttack, LinfPGDAttack, LinfMomentumIterativeAttack, \
    CarliniWagnerL2Attack, JacobianSaliencyMapAttack
from advertorch.attacks.utils import attack_whole_dataset
from models.ensemble import Ensemble
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count



class ProgressMeter(object):
	def __init__(self, num_batches, meters, prefix=""):
		self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
		self.meters = meters
		self.prefix = prefix

	def display(self, batch):
		entries = [self.prefix + self.batch_fmtstr.format(batch)]
		entries += [str(meter) for meter in self.meters]
		print('\t'.join(entries))

	def _get_batch_fmtstr(self, num_batches):
		num_digits = len(str(num_batches // 1))
		fmt = '{:' + str(num_digits) + 'd}'
		return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = args.lr * (0.1 ** (epoch // 30))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res



def init_logfile(filename: str, text: str):
	f = open(filename, 'w')
	f.write(text+"\n")
	f.close()


def log(filename: str, text: str):
	f = open(filename, 'a')
	f.write(text+"\n")
	f.close()


def init_logfile(filename: str, text: str):
	f = open(filename, 'w')
	f.write(text+"\n")
	f.close()

def log(filename: str, text: str):
	f = open(filename, 'a')
	f.write(text+"\n")
	f.close()


def requires_grad_(model:torch.nn.Module, requires_grad:bool) -> None:
	for param in model.parameters():
		param.requires_grad_(requires_grad)


def copy_code(outdir):
	"""Copies files to the outdir to store complete script with each experiment"""
	# embed()
	code = []
	exclude = set([])
	for root, _, files in os.walk("./code", topdown=True):
		for f in files:
			if not f.endswith('.py'):
				continue
			code += [(root,f)]

	for r, f in code:
		codedir = os.path.join(outdir,r)
		if not os.path.exists(codedir):
			os.mkdir(codedir)
		shutil.copy2(os.path.join(r,f), os.path.join(codedir,f))
	print("Code copied to '{}'".format(outdir))


def requires_grad_(model:torch.nn.Module, requires_grad:bool) -> None:
	for param in model.parameters():
		param.requires_grad_(requires_grad)



def Cosine(g1, g2):
	return torch.abs(F.cosine_similarity(g1, g2)).mean()  # + (0.05 * torch.sum(g1**2+g2**2,1)).mean()

def Magnitude(g1):
	return (torch.sum(g1**2,1)).mean() * 2


def gen_plot(transmat):
	import itertools
	plt.figure(figsize=(6, 6))
	plt.yticks(np.arange(0, 3, step=1))
	plt.xticks(np.arange(0, 3, step=1))
	cmp = plt.get_cmap('Blues')
	plt.imshow(transmat, interpolation='nearest', cmap=cmp, vmin=0, vmax=100.0)
	plt.title("Transfer attack success rate")
	plt.colorbar()
	thresh = 50.0
	for i, j in itertools.product(range(transmat.shape[0]), range(transmat.shape[1])):
		plt.text(j, i, "{:0.2f}".format(transmat[i, j]),
				 horizontalalignment="center",
				 color="white" if transmat[i, j] > thresh else "black")

	plt.ylabel('Target model')
	plt.xlabel('Base model')
	buf = io.BytesIO()
	plt.savefig(buf, format='jpeg')
	buf.seek(0)
	return buf

def evaltrans(args, loader, models, criterion, epoch, device, writer=None):

	for i in range(len(models)):
		models[i].eval()

	cos_losses_array = []
	list_of_combination = Combinator(args.num_models)
	for index, combination in list_of_combination:
		cos_losses_array.append(AverageMeter()) 

	for _, (inputs, targets) in enumerate(loader):

		inputs, targets = inputs.to(device), targets.to(device)
		batch_size = inputs.size(0)
		inputs.requires_grad = True
		grads = []
		for j in range(args.num_models):
			logits = models[j](inputs)
			loss = criterion(logits, targets)
			grad = autograd.grad(loss, inputs, create_graph=True)[0]
			grad = grad.flatten(start_dim=1)
			grads.append(grad)

		cos_loss = 0
		cos_array = []
		for combination in list_of_combination:
			cos_array.append(
				Cosine(grads[combination[0]], grads[combination[1]]))
		cos_loss = sum(cos_array) / len(cos_array) 
		for index, combination in list_of_combination:
			cos_losses_array[index].update(cos_array[index].item(), batch_size)
	adv = []
	for i in range(len(models)):
		curmodel = models[i]
		adversary = LinfPGDAttack(
			curmodel, loss_fn=criterion, eps=args.adv_eps,
			nb_iter=50, eps_iter=args.adv_eps / 10, rand_init=True, clip_min=0., clip_max=1.,
			targeted=False)
		adv.append(adversary)


	trans = np.zeros((len(models), len(models)))
	for i in range(len(models)):
		test_iter = tqdm(loader, desc='Batch', leave=False, position=2)
		_, label, pred, advpred = attack_whole_dataset(adv[i], test_iter, device="cuda")
		for j in range(len(models)):
			for r in range((_.size(0) - 1) // 200 + 1):
				inputc = _[r * 200: min((r + 1) * 200, _.size(0))]
				y = label[r * 200: min((r + 1) * 200, _.size(0))]
				__ = adv[j].predict(inputc)
				output = (__).max(1, keepdim=False)[1]
				trans[i][j] += (output == y).sum().item()
			trans[i][j] /= len(label)
			print(i, j, trans[i][j])

	plot_buf = gen_plot((1. - trans) * 100.)
	image = PIL.Image.open(plot_buf)
	image = ToTensor()(image)
	writer.add_image('TransferImage', image, epoch) 
	for index, combination in list_of_combination:
		writer.add_scalar('train/cos_{}'.format(combination),
							cos_losses_array[index].avg, epoch)


def evaltrans_robust_ensemble_attack(args, loader, models, criterion, epoch, device, writer=None):

	for i in range(len(models)):
		models[i].eval()

	cos_losses_array = []
	list_of_combination = Combinator(args.num_models)
	for index, combination in list_of_combination:
		cos_losses_array.append(AverageMeter()) 

	for _, (inputs, targets) in enumerate(loader):

		inputs, targets = inputs.to(device), targets.to(device)
		batch_size = inputs.size(0)
		inputs.requires_grad = True
		grads = []
		for j in range(args.num_models):
			logits = models[j](inputs)
			loss = criterion(logits, targets)
			grad = autograd.grad(loss, inputs, create_graph=True)[0]
			grad = grad.flatten(start_dim=1)
			grads.append(grad)

		cos_loss = 0
		cos_array = []
		for combination in list_of_combination:
			cos_array.append(
				Cosine(grads[combination[0]], grads[combination[1]]))
		cos_loss = sum(cos_array) / len(cos_array)
		for index, combination in list_of_combination:
			cos_losses_array[index].update(cos_array[index].item(), batch_size)
	# TODO: Ap dung thuat toan cho nay
	adv = []
	for i in range(len(models)):
		curmodel = models[i]
		adversary = LinfPGDAttack(
			curmodel, loss_fn=criterion, eps=args.adv_eps,
			nb_iter=50, eps_iter=args.adv_eps / 10, rand_init=True, clip_min=0., clip_max=1.,
			targeted=False)
		adv.append(adversary)
	
# # Assuming the following are defined:
# # f: your model
# # theta: parameters of your model
# # delta: function to compute the perturbation
# # loss_fn: your loss function
# # D: your dataset
# # lambdas: your lambda values

# # Initialize weights
# w = torch.randn((m,), requires_grad=True)

# # Define optimizer
# optimizer = optim.Adam([w], lr=0.01)

# # Optimization loop
# for epoch in range(num_epochs):
#     expected_loss = 0.0
#     for x, y in D:  # Iterate over the dataset
#         optimizer.zero_grad()
#         for i in range(m):  # Iterate over each f_i
#             # Compute the perturbed input
#             x_perturbed = x + delta(x, w)
#             # Compute the output of the model
#             output = f[i](x_perturbed, theta[i])
#             # Compute the loss
#             loss = lambdas[i] * loss_fn(output, y)
#             expected_loss += loss.item()
#         # Backward pass and optimization
#         expected_loss.backward()
#         optimizer.step()
#     print(f'Epoch {epoch+1}, Expected Loss: {expected_loss/len(D)}')

	trans = np.zeros((len(models), len(models)))
	for i in range(len(models)):
		test_iter = tqdm(loader, desc='Batch', leave=False, position=2)
		_, label, pred, advpred = attack_whole_dataset(adv[i], test_iter, device="cuda")
		for j in range(len(models)):
			for r in range((_.size(0) - 1) // 200 + 1):
				inputc = _[r * 200: min((r + 1) * 200, _.size(0))]
				y = label[r * 200: min((r + 1) * 200, _.size(0))]
				__ = adv[j].predict(inputc)
				output = (__).max(1, keepdim=False)[1]
				trans[i][j] += (output == y).sum().item()
			trans[i][j] /= len(label)
			print(i, j, trans[i][j])

	plot_buf = gen_plot((1. - trans) * 100.)
	image = PIL.Image.open(plot_buf)
	image = ToTensor()(image)
	writer.add_image('TransferImage', image, epoch)
	for index, combination in list_of_combination:
		writer.add_scalar('train/cos_{}'.format(combination),
							cos_losses_array[index].avg, epoch)



def test(loader, models, criterion, epoch, device, writer=None, print_freq=10, alpha = 1/3, required_alpha = False):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	end = time.time()

	# switch to eval mode
	for i in range(len(models)):
		models[i].eval()

	ensemble = Ensemble(models, alpha, required_alpha)
	with torch.no_grad():
		for i, (inputs, targets) in enumerate(loader):
			# measure data loading time
			data_time.update(time.time() - end)
			inputs, targets = inputs.to(device), targets.to(device)

			# compute output
			outputs = ensemble(inputs)
			loss = criterion(outputs, targets)

			# measure accuracy and record loss
			acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
			losses.update(loss.item(), inputs.size(0))
			top1.update(acc1.item(), inputs.size(0))
			top5.update(acc5.item(), inputs.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % print_freq == 0:
				print('Test: [{0}/{1}]\t'
					  'Time {batch_time.avg:.3f}\t'
					  'Data {data_time.avg:.3f}\t'
					  'Loss {loss.avg:.4f}\t'
					  'Acc@1 {top1.avg:.3f}\t'
					  'Acc@5 {top5.avg:.3f}'.format(
					i, len(loader), batch_time=batch_time, data_time=data_time,
					loss=losses, top1=top1, top5=top5))

		writer.add_scalar('loss/test', losses.avg, epoch)
		writer.add_scalar('accuracy/test@1', top1.avg, epoch)
		writer.add_scalar('accuracy/test@5', top5.avg, epoch)

def arr_to_str(x):
    M = len(x)
    x_str = "["
    for m in range(M-1):
        x_str += "{:.4f}, ".format(x[m])

    x_str += "{:.4f}]".format(x[M - 1])
    return x_str


def proj_onto_simplex(x):
	N = len(x)
	y_sorted, _ = torch.sort(x, descending=True)
	y = y_sorted # torch.sort(x)[::-1]
	rho = -1
	for i in range(N):
		q = y[i] + (1 - torch.sum(y[:i + 1])) / (1 + i)
		if q > 0:
			rho = i
	l = (1 - torch.sum(y[:rho+1])) / (rho + 1)
	x_hat = torch.zeros(N, device='cuda')
	for i in range(N):
		if x[i] + l > 0:
			x_hat[i] = x[i] + l
	return x_hat

def Combinator(length_models, combination_size = 2):
	return np.array(list(itertools.combinations(range(length_models), combination_size)))