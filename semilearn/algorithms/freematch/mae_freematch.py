import torch
import torch.nn.functional as F

from .utils import FreeMatchThresholdingHook
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool, mixup_one_target, mixup_one_target_withnoise
import numpy as np
import math

# TODO: move these to .utils or algorithms.utils.loss
def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val

def entropy_loss(mask, logits_s, prob_model, label_hist):
    mask = mask.bool()

    # select samples
    logits_s = logits_s[mask]

    prob_s = logits_s.softmax(dim=-1)
    _, pred_label_s = torch.max(prob_s, dim=-1)

    hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(logits_s.dtype)
    hist_s = hist_s / hist_s.sum()

    # modulate prob model 
    prob_model = prob_model.reshape(1, -1)
    label_hist = label_hist.reshape(1, -1)
    # prob_model_scaler = torch.nan_to_num(1 / label_hist, nan=0.0, posinf=0.0, neginf=0.0).detach()
    prob_model_scaler = replace_inf_to_zero(1 / label_hist).detach()
    mod_prob_model = prob_model * prob_model_scaler
    mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)

    # modulate mean prob
    mean_prob_scaler_s = replace_inf_to_zero(1 / hist_s).detach()
    # mean_prob_scaler_s = torch.nan_to_num(1 / hist_s, nan=0.0, posinf=0.0, neginf=0.0).detach()
    mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
    mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)

    loss = mod_prob_model * torch.log(mod_mean_prob_s + 1e-12)
    loss = loss.sum(dim=1)
    return loss.mean(), hist_s.mean()


@ALGORITHMS.register('maefreematch')
class MAEFreeMatch(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        self.init(T=args.T, hard_label=args.hard_label, ema_p=args.ema_p, use_quantile=args.use_quantile, clip_thresh=args.clip_thresh)
        self.lambda_e = args.ent_loss_ratio
        self.lambda_mae = args.mae_loss_ratio
        self.lambda_maecls = args.mae_cls_loss_ratio
        self.use_strongaug_mae = args.use_strongaug_mae
        self.mask_avg_pool = args.mask_cls_avg_pool
        self.mask_rate =args.mask_rate
        self.tau = args.tau

        self.mixup = args.mixup
        self.mixup_alpha = args.mixup_alpha
        self.lambda_mixed_loss = args.lambda_mixed_loss
        self.mix_noise =args.mix_noise
        self.lambda_sup_unsup = args.lambda_sup_unsup
        self.probmixup = args.probmixup


    def init(self, T, hard_label=True, ema_p=0.999, use_quantile=True, clip_thresh=False):
        self.T = T
        self.use_hard_label = hard_label
        self.ema_p = ema_p
        self.use_quantile = use_quantile
        self.clip_thresh = clip_thresh
        



    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FreeMatchThresholdingHook(num_classes=self.num_classes, momentum=self.args.ema_p, use_onethres=False),  "MaskingHook")
        super().set_hooks()


    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs, need_mae=True, mask_avg_pool=self.mask_avg_pool)
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
                outs_x_lb = self.model(x_lb, need_mae=True, mask_avg_pool=self.mask_avg_pool, mask_ratio=self.mask_rate) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                mae_logits_x_lb = outs_x_lb['mae_logits']

                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                # mae_logits_x_ulb_s = outs_x_ulb_s['mae_logits']
                # with torch.no_grad():

                outs_x_ulb_w = self.model(x_ulb_w, need_mae=True, mask_avg_pool=self.mask_avg_pool, mask_ratio=self.mask_rate)
                logits_x_ulb_w = outs_x_ulb_w['logits']
                feats_x_ulb_w = outs_x_ulb_w['feat']
                mae_logits_x_ulb_w = outs_x_ulb_w['mae_logits']

                # calculate mae part
                sup_mae_loss = outs_x_lb['mae_loss'].sum() / outs_x_lb['mae_mask'].sum()
                wasup_mae_loss = outs_x_ulb_w['mae_loss'].sum() / outs_x_ulb_w['mae_mask'].sum()
                # sasup_mae_loss = outs_x_ulb_s['mae_loss'].sum() / outs_x_ulb_s['mae_mask'].sum()
                mae_loss = sup_mae_loss + wasup_mae_loss

                if not math.isfinite(mae_loss.item()):
                    mae_loss = torch.Tensor([0]).to(x_lb.device)
                    print("Loss is {}, stopping training".format(mae_loss.item()))
                    # exit(-1)

            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}


            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            
            # stop gradient
            logits_x_ulb_w_de= logits_x_ulb_w.detach()
            # calculate mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb_w_de)


            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=logits_x_ulb_w_de,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T)
            
            # calculate unlabeled loss
            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                          pseudo_label,
                                          name='ce',
                                          mask=mask)
            
            # calculate mae logics loss
            mae_logics_sup_loss = self.ce_loss(mae_logits_x_lb/self.tau, y_lb, reduction='mean')
            mae_logics_unsup_loss = self.consistency_loss(mae_logits_x_ulb_w/self.tau,
                                        pseudo_label,
                                        'ce',
                                        mask=mask)
            # mas loss coff
            mae_loss_value = mae_loss.item()
            
            # calculate entropy loss
            if mask.sum() > 0:
               ent_loss, _ = entropy_loss(mask, logits_x_ulb_s, self.p_model, self.label_hist)
            else:
               ent_loss = 0.0

            assert not (self.mixup and self.probmixup), "mixup {} and probmixup {} cannot work together".format(self.mixup, self.probmixup)

            if self.mixup:
                # mixup
                # step 1. split clean and noise according to mask
                clean_x_ulb_s = x_ulb_s[mask!= 0]
                noise_x_ulb_s = x_ulb_s[mask == 0]
                clean_x_ulb_w = x_ulb_w[mask!= 0]
                noise_x_ulb_w = x_ulb_w[mask == 0]
                clean_pseudo_label_x_ulb = pseudo_label[mask !=0]
                noise_pseudo_label_x_ulb = pseudo_label[mask ==0]
                clean_pseudo_label_x_ulb = F.one_hot(clean_pseudo_label_x_ulb, self.num_classes)
                noise_pseudo_label_x_ulb = F.one_hot(noise_pseudo_label_x_ulb, self.num_classes)
                inputs = torch.cat([x_lb, clean_x_ulb_s, clean_x_ulb_w])
                noise_inputs = torch.cat([noise_x_ulb_s, noise_x_ulb_w])
                targets = torch.cat([F.one_hot(y_lb, self.num_classes),
                                     clean_pseudo_label_x_ulb,
                                     clean_pseudo_label_x_ulb])
                
                noise_targets = torch.cat([noise_pseudo_label_x_ulb, noise_pseudo_label_x_ulb])

                if self.mix_noise:
                    mixed_x, mixed_y, _ = mixup_one_target_withnoise(inputs, targets, noise_inputs, noise_targets, self.mixup_alpha, is_bias=True)
                else:
                    mixed_x, mixed_y, _ = mixup_one_target(inputs, targets, self.mixup_alpha, is_bias=True)

                # self.bn_controller.freeze_bn(self.model)
                mixed_output= self.model(mixed_x, need_mae=False)
                logits_mixed_output = mixed_output['logits']
                # self.bn_controller.unfreeze_bn(self.model)
                
                mixed_loss = self.ce_loss(logits_mixed_output, mixed_y, reduction='mean')
                mixed_loss_value = mixed_loss.item()
            elif self.probmixup:
                perm_order = np.random.permutation(x_ulb_s.size(0))
                shuffle_x_ulb_w = x_ulb_w[perm_order].clone()
                # print("shuffle_x_ulb_w", shuffle_x_ulb_w.size())
                prob, new_pseudo_label = torch.max(logits_x_ulb_w_de, dim=-1)
                # print("prob", prob.size())
                # print("new_pseudo_label", new_pseudo_label.size(), new_pseudo_label)
                lam_batch = prob / (prob + prob[perm_order])
                lam_batch = lam_batch.view(-1, 1, 1, 1)
                # print("lam_batch", lam_batch.size(), lam_batch)
                # print("(1 - lam_batch)", (1 - lam_batch).size(), (1 - lam_batch))
                new_prob = torch.maximum(prob, prob[perm_order])
                new_mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=new_prob, update = False)
                
                mixed_x = lam_batch * x_ulb_w + (1 - lam_batch) * shuffle_x_ulb_w

                mixed_output= self.model(mixed_x, need_mae=False)
                logits_mixed_output = mixed_output['logits']

                # mixed_loss = self.ce_loss(logits_mixed_output, new_pseudo_label, reduction='mean')
                mixed_loss = self.consistency_loss(logits_mixed_output,
                                          new_pseudo_label,
                                          name='ce',
                                          mask=new_mask)
                mixed_loss_value = mixed_loss.item()
            else:
                mixed_loss_value = mixed_loss = 0.0
            # choose_mae_mask = 1 - mask

            # if mae_mask.sum() > 0:
            #     # print("mae_loss", outs_x_ulb_w['mae_loss'].sum(dim=-1))
            #     # print("mae_mask", outs_x_ulb_w['mae_mask'].sum(dim=-1))
            #     masked_mae_loss= (outs_x_ulb_w['mae_loss'].sum(dim=-1) / outs_x_ulb_w['mae_mask'].sum(dim=-1) * mae_mask).mean()
            # else:
            #     masked_mae_loss = torch.tensor([0])
            # if choose_mae_mask.sum() > 0:
            #     masked_mae_loss = (outs_x_ulb_w['mae_loss'].sum(dim=-1) / outs_x_ulb_w['mae_mask'].sum(dim=-1) * choose_mae_mask).sum() / choose_mae_mask.sum()
            # else:
            #     masked_mae_loss = choose_mae_mask.sum()
            # + self.lambda_mixed_loss * mixed_loss
            total_loss = self.lambda_sup_unsup * (sup_loss + self.lambda_u * unsup_loss) + self.lambda_e * ent_loss + \
                         self.lambda_mixed_loss * mixed_loss + \
                         self.lambda_maecls * (mae_logics_sup_loss + mae_logics_unsup_loss) + \
                         self.lambda_mae * mae_loss 

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(),
                                         lambda_sup_unsup = self.lambda_sup_unsup,
                                         mae_logics_sup_loss =mae_logics_sup_loss.item(),
                                         mae_logics_unsup_loss = mae_logics_unsup_loss.item(),
                                         mixed_loss= mixed_loss_value,
                                        #  masked_mae_loss = masked_mae_loss.item(),
                                        #  ent_loss = ent_loss.item(),
                                         mae_loss=mae_loss.item(),
                                         lambda_mae = self.lambda_mae,
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['p_model'] = self.hooks_dict['MaskingHook'].p_model.cpu()
        save_dict['time_p'] = self.hooks_dict['MaskingHook'].time_p.cpu()
        save_dict['label_hist'] = self.hooks_dict['MaskingHook'].label_hist.cpu()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['MaskingHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].time_p = checkpoint['time_p'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].label_hist = checkpoint['label_hist'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--ent_loss_ratio', float, 0.01),
            SSL_Argument('--use_quantile', str2bool, False),
            SSL_Argument('--clip_thresh', str2bool, False),
            SSL_Argument('--mixup', str2bool, True),
            SSL_Argument('--mix_noise', str2bool, True),
            SSL_Argument('--mixup_alpha', float, 0.5),
            SSL_Argument('--lambda_mixed_loss', float, 0.95),
            SSL_Argument('--mask_rate', float, 0.3),
            SSL_Argument('--tau', float, 10),
            SSL_Argument('--lambda_sup_unsup', float, 1.0),
            SSL_Argument('--probmixup', str2bool, False),
            SSL_Argument('--decoder_depth', int, 4),
            SSL_Argument('--decoder_width', int, 192),
            
        ]