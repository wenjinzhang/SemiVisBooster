import torch
import torch.nn.functional as F

from .msn_losses import init_msn_loss

from .utils import FreeMatchThresholdingHook
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool


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

def one_hot(targets, num_classes, smoothing=0.0):
        off_value = smoothing / num_classes
        on_value = 1. - smoothing + off_value
        # targets = targets.long().view(-1, 1).to(device)
        targets = targets.long().view(-1, 1)
        return torch.full((len(targets), num_classes), off_value, device=targets.device).scatter_(1, targets, on_value)


@ALGORITHMS.register('msnfreematch')
class MSNFreeMatch(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        self.init(T=args.T, hard_label=args.hard_label, ema_p=args.ema_p, use_quantile=args.use_quantile, clip_thresh=args.clip_thresh)
        self.lambda_e = args.ent_loss_ratio
        self.patch_drop = args.patch_drop
        self.memax_weight = args.memax_weight
        self.ent_weight = args.ent_weight
        
         # -- make prototypes
        self.prototypes, self.proto_labels = None, None
        if args.num_proto > 0:
            with torch.no_grad():
                self.prototypes = torch.empty(args.num_proto, args.output_dim)
                _sqrt_k = (1./args.output_dim)**0.5
                torch.nn.init.uniform_(self.prototypes, -_sqrt_k, _sqrt_k)
                self.prototypes = torch.nn.parameter.Parameter(self.prototypes).cuda(self.gpu)

                # -- init prototype labels
                self.proto_labels = one_hot(torch.tensor([i for i in range(args.num_proto)]), args.num_proto).cuda(self.gpu)

            if not args.freeze_proto:
                self.prototypes.requires_grad = True
            logger.info(f'Created prototypes: {self.prototypes.shape}')
            logger.info(f'Requires grad: {self.prototypes.requires_grad}')
        
        # -- init losses
        self.msn = init_msn_loss(
            num_views=args.focal_views + args.rand_views,
            tau=args.temperature,
            me_max=args.me_max,
            return_preds=True)
        
        
        # add prototype as updatable parameters
        if self.prototypes is not None:
            params_group_dict = {
                'params': [self.prototypes],
                'LARS_exclude': True,
                'WD_exclude': True,
                'weight_decay': 0
            }
            ## add one more params to optimizer
            self.optimizer, self.scheduler = self.set_optimizer(additional_params = params_group_dict)

    def init(self, T, hard_label=True, ema_p=0.999, use_quantile=True, clip_thresh=False):
        self.T = T
        self.use_hard_label = hard_label
        self.ema_p = ema_p
        self.use_quantile = use_quantile
        self.clip_thresh = clip_thresh

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FreeMatchThresholdingHook(num_classes=self.num_classes, momentum=self.args.ema_p), "MaskingHook")
        super().set_hooks()


    def train_step(self, x_lb, x_lb_msn, y_lb, x_ulb_w, x_ulb_s, x_ulb_msn):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                exit() # disable this branch
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                ## msn loss
                # anchor view
                # print("x_lb_msn_msn info", type(x_lb_msn), "lens of x_lb_msn",len(x_lb_msn), x_lb_msn[0].size())
                # print("x_ulb_msn info", type(x_ulb_msn), "lens of x_ulb_msn",len(x_ulb_msn), x_ulb_msn[0].size())
                msn_inputs = [torch.cat((lb, ulb)) for (lb, ulb) in zip(x_lb_msn, x_ulb_msn)]

                # print("msn_inputs info", type(msn_inputs), "lens of msn_inputs",len(msn_inputs), msn_inputs[0].size())
                h, z = self.model(msn_inputs[1:], msn=True, patch_drop=self.patch_drop)
                with torch.no_grad():
                    # target view
                    h, _ = self.ema_model(msn_inputs[0], msn=True)

                # Step 1. convert representations to fp32
                h, z = h.float(), z.float()

                # Step 2. determine anchor views/supports and their
                #         corresponding target views/supports
                anchor_views, target_views = z, h.detach()

                # Step 3. compute msn loss with me-max regularization
                (ploss, me_max, ent, logs, targets, probs) = self.msn(
                    T=0.25,
                    use_sinkhorn=True,
                    use_entropy=True,
                    anchor_views=anchor_views,
                    target_views=target_views,
                    proto_labels=self.proto_labels,
                    prototypes=self.prototypes)
                
                temp_predict = self.model.module.msn_predict_head(probs)
                
                num_views= (len(msn_inputs) - 1)
                len_lab =  num_views * y_lb.size(0)
                labeled_predict = temp_predict[:len_lab,:]
                
                labels_y = y_lb.repeat_interleave(num_views)
                
                msn_sup_loss = self.ce_loss(labeled_predict, labels_y, reduction='mean')
                
                msn_loss = ploss + self.memax_weight * me_max + self.ent_weight * ent

                outs_x_lb = self.model(x_lb)
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s[:x_lb.size(0)])
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w[:x_lb.size(0)])
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
                
                
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}

            
            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            # calculate mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb_w)


            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=logits_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T)
            
            # calculate unlabeled loss
            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                          pseudo_label,
                                          'ce',
                                          mask=mask)
            
            # calculate entropy loss
            if mask.sum() > 0:
               ent_loss, _ = entropy_loss(mask, logits_x_ulb_s, self.p_model, self.label_hist)
            else:
               ent_loss = 0.0
            # ent_loss = 0.0
            total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_e * ent_loss + msn_loss  + msn_sup_loss

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(),
                                         msn_loss = msn_loss.item(),
                                         msn_sup_loss = msn_sup_loss.item(),
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
            SSL_Argument('--patch_drop', float, 0.15),
        ]