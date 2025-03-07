import torch
import torch.nn.functional as F

from .utils import FreeMatchThresholdingHook
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool

from transformers import CLIPModel, CLIPProcessor, BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel, \
      T5Tokenizer, T5EncoderModel, DebertaV2Tokenizer, DebertaV2Model


import torch.nn as nn
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class VisionProjection(nn.Module):
    def __init__(self, input_dim=384, output_dim=512):
        super(VisionProjection, self).__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, vision_embeds):
        return self.proj(vision_embeds)
    
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


# Define the InfoNCE loss function
def info_nce_loss(vision_embeds, text_embeds, temperature=0.07):
    # Normalize the embeddings
    vision_embeds = F.normalize(vision_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    
    # Compute the similarity matrix between vision and text embeddings
    logits = torch.matmul(vision_embeds, text_embeds.T) / temperature
    
    # Labels are simply indices (0...N) assuming a 1:1 correspondence between embeddings
    labels = torch.arange(len(vision_embeds)).to(vision_embeds.device)
    
    # Compute cross entropy loss using the similarity matrix
    loss = F.cross_entropy(logits, labels)
    return loss


# import torch
# import torch.nn.functional as F

def compute_contrastive_loss(vision_embeds, text_embeds,  labels, temperature=0.07, num_classes=None):
    """
    Compute class-constrained contrastive loss between vision embeddings and text embeddings.

    Parameters:
    - vision_embeds (torch.Tensor): A batch of visual embeddings (batch_size x embed_dim).
    - text_embeds (torch.Tensor): A batch of text embeddings (batch_size x embed_dim).
    - temperature (float): The temperature scaling factor for the contrastive loss.
    - labels (torch.Tensor): A tensor of shape (batch_size,) containing the class labels for each sample.
    - num_classes (int): The number of constrained classes used in the contrastive loss.

    Returns:
    - torch.Tensor: The computed contrastive loss.
    """
    # Normalize the embeddings for better numerical stability
    vision_embeds = F.normalize(vision_embeds, p=2, dim=-1)
    text_embeds = F.normalize(text_embeds, p=2, dim=-1)

    # Compute similarity matrix (dot product between vision and text embeddings)
    similarity_matrix = torch.matmul(vision_embeds, text_embeds.T) / temperature
    
    # Create mask for positive samples (diagonal elements for matching vision-text pairs)
    labels = labels.view(-1, 1)  # Reshape to column vector
    mask = torch.eq(labels, labels.T).float()  # Mask where 1 means same class, 0 means different class
    
    # Positive pairs (vision embedding and corresponding text embedding of the same class)
    pos_pairs = torch.exp(similarity_matrix) * mask
    
    # Sum of exponential similarities (both positive and negative samples)
    exp_similarity_matrix = torch.exp(similarity_matrix)
    neg_pairs_sum = exp_similarity_matrix.sum(dim=-1) - pos_pairs.sum(dim=-1)  # Exclude the positives from the sum
    
    # Compute the contrastive loss using NT-Xent formula
    contrastive_loss = -torch.log((pos_pairs.sum(dim=-1)) / (pos_pairs.sum(dim=-1) + neg_pairs_sum))
    
    # Return the mean loss across the batch
    return contrastive_loss.mean()

# def orthogonal_regularization(text_embeds, margin=1e-5):
#     """
#     Encourage text embeddings to be orthogonal (dissimilar) by minimizing 
#     the absolute value of their pairwise dot products.
#     """
#     # Normalize the text embeddings
#     text_embeds = F.normalize(text_embeds, dim=-1)
#     # Compute pairwise dot product between all text embeddings
#     similarity_matrix = torch.matmul(text_embeds, text_embeds.T)
#     # Create a mask to ignore the diagonal (self-similarity)
#     identity_mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)
#     # Compute the orthogonality loss (minimize off-diagonal similarity)
#     orthogonality_loss = torch.abs(similarity_matrix * (1 - identity_mask)).mean()
#     # Penalize the similarity if it's above a small margin (enforcing dissimilarity)
#     orthogonality_loss = torch.clamp(orthogonality_loss - margin, min=0.0)
    
#     return orthogonality_loss

def orthogonal_regularization(embeds, labels, margin=1e-5):
    """
    Encourage embeddings to be orthogonal (dissimilar) for different classes by minimizing 
    the absolute value of their pairwise dot products, except for embeddings of the same class.
    
    Parameters:
    - embeds: The projected vision embeddings.
    - labels: The pseudo labels for the embeddings.
    - margin: The margin to penalize high similarity.
    
    Returns:
    - orthogonality_loss: The computed orthogonal regularization loss.
    """
    # Normalize the embeddings for numerical stability
    embeds = F.normalize(embeds, dim=-1)

    # Compute similarity matrix (dot product between embeddings)
    similarity_matrix = torch.matmul(embeds, embeds.T)

    # Create mask where 1 means different class and 0 means same class
    labels = labels.view(-1, 1)  # Reshape labels to be column vector
    class_mask = (labels != labels.T).float()  # Mask where 1 indicates different classes

    # Ignore the diagonal (self-similarity) by setting the diagonal to 0
    identity_mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)
    class_mask = class_mask * (1 - identity_mask)  # Combine the class mask with identity mask
    
    # Compute orthogonal loss by penalizing similarity for different class embeddings
    orthogonality_loss = torch.abs(similarity_matrix * class_mask).mean()
    
    # Apply margin to penalize high similarity
    orthogonality_loss = torch.clamp(orthogonality_loss - margin, min=0.0)
    
    return orthogonality_loss

@ALGORITHMS.register('clipfreematch')
class CLIPFreeMatch(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        self.vt_proj = args.vt_proj
        self.text_model = args.text_model
        self.long_text = args.long_text
        self.init_text_embeding(args=args)

        super().__init__(args, net_builder, tb_log, logger) 
        self.init(T=args.T, hard_label=args.hard_label, ema_p=args.ema_p, use_quantile=args.use_quantile, clip_thresh=args.clip_thresh)
        self.lambda_e = args.ent_loss_ratio
        self.lambda_c = args.lambda_c
        self.lambda_ortho = args.lambda_ortho
        self.contras_tempeture = args.contras_tempeture
        
    def init_text_embeding(self, args):
        from semilearn.datasets import cifar100_label_name, cifar100_label_text, food101_label_name, food101_label_text, \
                tissuemnist_label_name, tissuemnist_label_text
        if args.dataset == 'cifar100':
            label_name = cifar100_label_text if self.long_text else cifar100_label_name
        elif args.dataset == 'food101':
            label_name = food101_label_text if self.long_text else food101_label_name
        elif args.dataset == 'tissuemnist':
            label_name = tissuemnist_label_text if self.long_text else tissuemnist_label_name
        elif args.dataset == 'stl10':
            from semilearn.datasets import stl10_label_name, stl10_label_text
            label_name = stl10_label_text if self.long_text else stl10_label_name
        elif args.dataset == 'semi_aves':
            from semilearn.datasets import semi_aves_label_name, semi_aves_label_text
            label_name = semi_aves_label_text if self.long_text else semi_aves_label_name
        else:
            raise Exception(f"Unknown label name for dataset {args.dataset}")

        label_texts = [f"a photo of a {label}" for label in label_name.values()]
        print("================Using lable_name as the text supervision================")
        print(label_name)
        print("========================================================================")

        if self.text_model == "clip":
            # CLIP Model
            print("using clip as text encoder!")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            label_name_inputs = self.clip_processor(text=label_texts, return_tensors="pt", padding=True)
            label_name_inputs = {k: v.cuda() for k, v in label_name_inputs.items()}
            
            with torch.no_grad():
                self.text_embeddings_bank = self.clip_model.get_text_features(**label_name_inputs)

        elif self.text_model == "bert":
            # BERT Model
            print("using bert as text encoder!")
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            model = BertModel.from_pretrained("bert-base-uncased").cuda()
            label_name_inputs = tokenizer(label_texts, return_tensors="pt", padding=True, truncation=True)
            label_name_inputs = {k: v.cuda() for k, v in label_name_inputs.items()}

            with torch.no_grad():
                outputs = model(**label_name_inputs)
                self.text_embeddings_bank = outputs.last_hidden_state.mean(dim=1)  # Pool embeddings

        elif self.text_model == "distilbert":
            # DistilBERT Model
            print("using distilbert as text encoder!")
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            model = DistilBertModel.from_pretrained("distilbert-base-uncased").cuda()
            label_name_inputs = tokenizer(label_texts, return_tensors="pt", padding=True, truncation=True)
            label_name_inputs = {k: v.cuda() for k, v in label_name_inputs.items()}

            with torch.no_grad():
                outputs = model(**label_name_inputs)
                self.text_embeddings_bank = outputs.last_hidden_state.mean(dim=1)  # Pool embeddings

        elif self.text_model == "t5":
            # T5 Model (Encoder-only)
            print("using t5 as text encoder!")
            tokenizer = T5Tokenizer.from_pretrained("t5-base")
            model = T5EncoderModel.from_pretrained("t5-base").cuda()
            label_name_inputs = tokenizer(label_texts, return_tensors="pt", padding=True, truncation=True)
            label_name_inputs = {k: v.cuda() for k, v in label_name_inputs.items()}

            with torch.no_grad():
                outputs = model(**label_name_inputs)
                self.text_embeddings_bank = outputs.last_hidden_state.mean(dim=1)  # Pool embeddings

        elif self.text_model == "deberta":
            # DeBERTa Model (DebertaV2)
            tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v2-xlarge")
            model = DebertaV2Model.from_pretrained("microsoft/deberta-v2-xlarge").cuda()
            label_name_inputs = tokenizer(label_texts, return_tensors="pt", padding=True, truncation=True)
            label_name_inputs = {k: v.cuda() for k, v in label_name_inputs.items()}

            with torch.no_grad():
                outputs = model(**label_name_inputs)
                self.text_embeddings_bank = outputs.last_hidden_state.mean(dim=1)  # Pool embeddings

        else:
            raise Exception(f"unimplemented text_model: {self.text_model}")

        print("<<<<<<<================================>>>>>")
        print("self.text_embeddings_bank.size()", self.text_embeddings_bank.size())
        self.text_embed_dim = self.text_embeddings_bank.size(-1)


    def init(self, T, hard_label=True, ema_p=0.999, use_quantile=True, clip_thresh=False):
        self.T = T
        self.use_hard_label = hard_label
        self.ema_p = ema_p
        self.use_quantile = use_quantile
        self.clip_thresh = clip_thresh

        # label_texts = [f"a photo of a {label}" for label in cifar100_labels.values()]
        # self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
        # self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # label_name_inputs = self.clip_processor(text=label_texts, return_tensors="pt", padding=True)
        # label_name_inputs = {k: v.cuda() for k, v in label_name_inputs.items()}
        # with torch.no_grad():  # Disable gradient computation
        #     self.text_embeddings_bank = self.clip_model.get_text_features(**label_name_inputs)


    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FreeMatchThresholdingHook(num_classes=self.num_classes, momentum=self.args.ema_p), "MaskingHook")
        super().set_hooks()


    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]
        device = x_lb.device
        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
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

            

            vision_embeds_x_lb = feats_x_lb
            text_embeds_x_lb = self.text_embeddings_bank[y_lb]
            
            
            # Apply the mask to filter out the embeddings where mask == 1
            vision_embeds_ulb_s = feats_x_ulb_s[mask.bool()]
            vision_embeds_ulb_w = feats_x_ulb_w[mask.bool()]
            text_embeds_ulb_w = self.text_embeddings_bank[pseudo_label[mask.bool()]]
            labels = pseudo_label[mask.bool()]

            vision_embeds = torch.cat([vision_embeds_x_lb, vision_embeds_ulb_w, vision_embeds_ulb_s])
            text_embeds = torch.cat([text_embeds_x_lb, text_embeds_ulb_w, text_embeds_ulb_w])
            
            labels = torch.cat([y_lb, labels, labels])
            
            # Project vision embeddings to match the dimensionality of text embeddings
            vision_embeds = self.model.module.projector(vision_embeds, text_embeds)

            # Compute the InfoNCE contrastive loss between masked vision and text embeddings
            # contrastive_loss = info_nce_loss(vision_embeds, text_embeds)
            contrastive_loss= compute_contrastive_loss(vision_embeds=vision_embeds,
                                        text_embeds=text_embeds,
                                        labels=labels, 
                                        temperature=self.contras_tempeture)
            
            contrastive_loss_value = contrastive_loss.item()

            # Apply orthogonal regularization using pseudo labels
            ortho_loss = orthogonal_regularization(vision_embeds, labels)

            total_loss = sup_loss + self.lambda_u * unsup_loss + \
                        self.lambda_e * ent_loss + self.lambda_c * contrastive_loss +\
                        ortho_loss * self.lambda_ortho

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(),
                                         contrastive_loss = contrastive_loss_value,
                                         ortho_loss = ortho_loss.item(),
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
            SSL_Argument('--lambda_c', float, 0.1),
            SSL_Argument('--contras_tempeture', float, 0.07),
            SSL_Argument('--lambda_ortho', float, 0.01),
            SSL_Argument('--vt_proj', str, 'linear'),
            SSL_Argument('--text_model', str, 'clip'),
            SSL_Argument('--long_text', str2bool, True)
        ]