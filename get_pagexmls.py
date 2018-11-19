import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse
import pagexml
import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer


assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))



def labels_to_text(labels,alphabet):
    ret = []
    for c in labels:
	if c ==0:# len(alphabet):  # CTC Blank
	    ret.append("")
	else:
	    ret.append(alphabet[c])
    return "".join(ret)
    out = np.reshape(out,(1,w,len(self.alphabet)+1))
    ret=[]	
    for j in range(out.shape[0]):
	out_best = list(np.argmax(out[j, 2:], 1))
	out_best = [k for k, g in itertools.groupby(out_best)]
	outstr = labels_to_text(out_best)
	ret.append(outstr)
	
    return ret

def main(args=None):
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--coco_path', help='Path to COCO directory')
	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
	parser.add_argument('--score_threshold', help='Score above which boxes are kept',default=0.5)

	parser.add_argument('--model', help='Path to model (.pt) file.')

	parser = parser.parse_args(args)

	if parser.dataset == 'coco':
		dataset_val = CocoDataset(parser.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))
	elif parser.dataset == 'csv':
		dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

	sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
	dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

	retinanet = torch.load(parser.model)
	score_threshold = float(parser.score_threshold)
	alphabet = " abcdefghijklmnopqrstuvwxy z"
	use_gpu = True

	if use_gpu:
		retinanet = retinanet.cuda()

	retinanet.eval()

	unnormalize = UnNormalizer()

	def draw_caption(image, box, caption):

		b = np.array(box).astype(int)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

	pxml = pagexml.PageXML()
	if not os.path.exists('pagexmls'):
		os.mkdir('pagexmls')
	for idx, data in enumerate(dataloader_val):
		
		# Create a new page xml
		image_name = str(idx)+'.jpg'
		file ='pagexmls/'+image_name
		
		gtxml_name = os.path.join(image_name.split('/')[-1].split('.')[-2])


		with torch.no_grad():
			st = time.time()
			#scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
			im=data['img']
			im = im.cuda().float()
			scores, classification, transformed_anchors = retinanet(im)
			print('Elapsed time: {}'.format(time.time()-st))
			idxs = np.where(scores>score_threshold)
			img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()
			#img = np.array(255 * unnormalize(im)).copy()

			img[img<0] = 0
			img[img>255] = 255

			img = np.transpose(img, (1, 2, 0))

			img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
			width = img.shape[1]
			height = img.shape[0]
			cv2.imwrite(file,img)
			'''scale_x = float(original_w)/width
			scale_y = float(original_h)/height
			print "scales x y",scale_x,scale_y'''

			pxml.newXml('retinanet_dets',image_name,width,height)
			for j in range(idxs[0].shape[0]):

				# Initialize object for setting confidence values
				conf = pagexml.ptr_double()
				bbox = transformed_anchors[idxs[0][j], :]
				x1 = int(bbox[0])
				y1 = int(bbox[1])
				x2 = int(bbox[2])
				y2 = int(bbox[3])
				label_name = dataset_val.labels[int(classification[idxs[0][j]])]
				draw_caption(img, (x1, y1, x2, y2), label_name)

				cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
				
				# Add a text region to the Page
				page = pxml.selectNth("//_:Page",0)
				reg = pxml.addTextRegion(page)

				# Set text region bounding box with a confidence
				conf.assign(1)
				pxml.setCoordsBBox( reg,x1, y1, x2-x1, y2-y1, conf )
				
				transcripts=[]
				confs=[]
				seq_len = int(bbox[4])
				for k in range(seq_len):
					transcripts.append(np.argmax(bbox[(5+k*27):((5+(k+1)*27))]))
				transcripts=np.array(transcripts)
				transcript=labels_to_text(transcripts,alphabet)
				pxml.setTextEquiv( reg, "".join([alphabet[transcripts[k]] for k in range(len(transcripts))]), conf )
	

				# Set the text for the text region
				conf.assign(0.9)

				# Add property to text region
				pxml.setProperty( reg, "key", "value" )

				# Add a second page with a text region and specific id
				#page = pxml.addPage("example_image_2.jpg", 300, 300)
				#reg = pxml.addTextRegion( page, "regA" )
				#pxml.setCoordsBBox( reg, 15, 12, 76, 128 )

			# Write XML to file
			pxml.write('pagexmls/'+gtxml_name+".xml")
			'''cv2.imshow('img', img)
			cv2.waitKey(0)'''
			print "Get more preds?"
			continue_eval =raw_input()
			if continue_eval!='n' and continue_eval!='N': continue

if __name__ == '__main__':
 main()
