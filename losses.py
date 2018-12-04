import numpy as np
import torch
import torch.nn as nn
import pdb
torch.set_printoptions(threshold=5000)

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


class FocalLoss(nn.Module):
    #def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations,criterion,transcription,selected_indices):
        alpha = 0.25
        gamma = 2.0
	alphabet_len = 27
	seq_len = 60
	max_label_len = 20
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []
	#regressions = regressions[...,:4]
        anchor = anchors[0, :, :]
        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

        for j in range(batch_size):
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
	    
            if bbox_annotation.shape[0] == 0:
                regression_losses.append(torch.tensor(0).float().cuda())
                classification_losses.append(torch.tensor(0).float().cuda())

                continue

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations

            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1


            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1
            targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones(targets.shape).cuda() * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression
            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

		
                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                #targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh,label_lengths))
                targets = targets.t()

                targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                #targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2,1]]).cuda()


                negative_indices = 1 - positive_indices

                regression_diff = torch.abs(targets - regression[positive_indices, :4])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
		
		if selected_indices.shape[0]>250:
			# compute ctc loss
			transcript_labels = bbox_annotation[IoU_argmax,5:(5+max_label_len)][selected_indices,:].int().cpu()
			label_lengths = torch.sum(transcript_labels>0,dim=1).int()
			#transcript_labels = assigned_annotations[selected_indices,5:(5+max_label_len)]
			transcript_labels = transcript_labels[transcript_labels>0]
			transcript_labels = torch.clamp(transcript_labels,1,alphabet_len)
			transcript_labels = transcript_labels.view(transcript_labels.numel()).cpu()
		
		
			transcription = transcription.view(-1,seq_len,alphabet_len).transpose(0,1).contiguous()
			transcription.requires_grad_(True)
			
			probs_sizes = torch.IntTensor(())
			probs_sizes = probs_sizes.new_full((transcription.shape[1],),seq_len)
			ctc_loss = criterion(transcription,transcript_labels,probs_sizes,label_lengths)
			ctc_loss = ctc_loss.float()
			ctc_loss = (ctc_loss/label_lengths.shape[0])
			print label_lengths.shape[0]
			ctc_loss = ctc_loss.cuda()
		else: ctc_loss = torch.tensor(1).float().cuda()
	        '''# compute ctc loss
		transcript_labels = assigned_annotations[:,5:(5+max_label_len)].int().cpu()
		label_lengths = torch.sum(transcript_labels>0,dim=1).int()
		transcript_preds = regressions[j,positive_indices,5:]
		transcript_preds = transcript_preds.view(-1,40,alphabet_len).transpose(0,1).contiguous()
		transcript_labels = transcript_labels[transcript_labels>0]
		transcript_labels = torch.clamp(transcript_labels,1,alphabet_len)
		transcript_labels = transcript_labels.view(transcript_labels.numel())
				
		#label_lengths = label_lengths.new_full((transcript_preds.shape[1],),max_label_len)
		probs_sizes = torch.IntTensor(())
		probs_sizes = probs_sizes.new_full((transcript_preds.shape[1],),max_seq_len)
		transcript_preds.requires_grad_(True)
		#print "CTC input shapes (probs,labels,label lens,prob sizes):",transcript_preds.shape,transcript_labels.shape,label_lengths.shape,probs_sizes.shape
		#ctc_loss = criterion(transcript_preds,transcript_labels,label_lengths,probs_sizes)
		ctc_loss = criterion(transcript_preds,transcript_labels,label_lengths,label_lengths)
		ctc_loss = ctc_loss.float()
		ctc_loss = (ctc_loss/label_lengths.shape[0]).cuda()
		#print "CTC LOSS Value",ctc_loss
		#print "class loss",torch.stack(classification_losses).mean(dim=0, keepdim=True)'''
            else:
                regression_losses.append(torch.tensor(0).float().cuda())
        return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True),ctc_loss

    
