# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch


from .utils import AdaMatchThresholdingHook
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, DistAlignEMAHook
from semilearn.algorithms.utils import SSL_Argument, str2bool, mixup_one_target


@ALGORITHMS.register('maeadamatch')
class MAEAdaMatch(AlgorithmBase):
    """
        AdaMatch algorithm (https://arxiv.org/abs/2106.04732).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
            - ema_p (`float`):
                momentum for average probability
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        self.init(p_cutoff=args.p_cutoff, T=args.T, hard_label=args.hard_label, ema_p=args.ema_p)

        self.lambda_mae = args.mae_loss_ratio
        self.lambda_maecls = args.mae_cls_loss_ratio
        self.use_strongaug_mae = args.use_strongaug_mae
        self.mask_avg_pool = args.mask_cls_avg_pool

    def init(self, p_cutoff, T, hard_label=True, ema_p=0.999, mixup_alpha=0.5):
        self.p_cutoff = p_cutoff
        self.T = T
        self.use_hard_label = hard_label
        self.ema_p = ema_p
        self.mixup_alpha = mixup_alpha


    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(
            DistAlignEMAHook(num_classes=self.num_classes, momentum=self.args.ema_p, p_target_type='model'), 
            "DistAlignHook")
        self.register_hook(AdaMatchThresholdingHook(), "MaskingHook")
        super().set_hooks()


    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs, need_mae=True)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)

                # calculate reconstract loss(mae loss)
                len_img = x_lb.shape[0] + x_ulb_w.shape[1] 
                if self.use_strongaug_mae:
                    mae_loss = outputs['mae_loss'].sum() / outputs['mae_mask'].sum()
                else:
                    mae_loss = outputs['mae_loss'][:len_img].sum() / outputs['mae_mask'][:len_img].sum()

            else:
                outs_x_lb = self.model(x_lb, need_mae=True, mask_avg_pool=self.mask_avg_pool) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                mae_logits_x_lb = outs_x_lb['mae_logits']

                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']

                # with torch.no_grad():
                outs_x_ulb_w = self.model(x_ulb_w, need_mae=True, mask_avg_pool=self.mask_avg_pool)
                logits_x_ulb_w = outs_x_ulb_w['logits']
                feats_x_ulb_w = outs_x_ulb_w['feat']
                mae_logits_x_ulb_w = outs_x_ulb_w['mae_logits']

                # calculate mae part
                sup_mae_loss = outs_x_lb['mae_loss'].sum() / outs_x_lb['mae_mask'].sum()
                wasup_mae_loss = outs_x_ulb_w['mae_loss'].sum() / outs_x_ulb_w['mae_mask'].sum()
                # sasup_mae_loss = outs_x_ulb_s['mae_loss'].sum() / outs_x_ulb_s['mae_mask'].sum()
                mae_loss = sup_mae_loss + wasup_mae_loss


            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}
                    

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            # probs_x_lb = torch.softmax(logits_x_lb.detach(), dim=-1)
            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w.detach(), dim=-1)
            probs_x_lb = self.compute_prob(logits_x_lb.detach())
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())

            # distribution alignment 
            probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w, probs_x_lb=probs_x_lb)

            # calculate weight
            mask = self.call_hook("masking", "MaskingHook", logits_x_lb=probs_x_lb, logits_x_ulb=probs_x_ulb_w, softmax_x_lb=False, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)

            # calculate loss
            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                               pseudo_label,
                                               'ce',
                                               mask=mask)
            
            # calculate mae logics loss
            mae_logics_sup_loss = self.ce_loss(mae_logits_x_lb, y_lb, reduction='mean')
            mae_logics_unsup_loss = self.consistency_loss(mae_logits_x_ulb_w,
                                        pseudo_label,
                                        'ce',
                                        mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_mae * mae_loss + self.lambda_maecls*(mae_logics_sup_loss+ mae_logics_unsup_loss)

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(),
                                         mae_logics_sup_loss = mae_logics_sup_loss.item(),
                                         mae_logics_unsup_loss = mae_logics_unsup_loss.item(),
                                         mae_loss = mae_loss.item(),
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict


    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu()
        save_dict['p_target'] = self.hooks_dict['DistAlignHook'].p_target.cpu()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['DistAlignHook'].p_target = checkpoint['p_target'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]