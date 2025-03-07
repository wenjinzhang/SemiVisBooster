import torch
import torch.nn.functional as F
import numpy as np

from .utils import FreeMatchThresholdingHook
from .utils import mixup_data, cutmix_data, rand_bbox
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool

from transformers import CLIPModel, CLIPProcessor, BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel, \
      T5Tokenizer, T5EncoderModel, DebertaV2Tokenizer, DebertaV2Model
from sentence_transformers import SentenceTransformer


import torch.nn as nn
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

def sentence_to_vector(sentence, ft_model):
    """
    Convert a sentence into a vector using FastText embeddings with mean pooling.
    
    Args:
        sentence (str): The input sentence.
        ft_model: The FastText model.

    Returns:
        np.array: A fixed-size sentence vector.
    """
    words = sentence.split()  # Tokenize sentence
    word_vectors = [ft_model.get_word_vector(word) for word in words if word in ft_model]

    if not word_vectors:  # If no words found in FastText vocabulary
        return np.zeros(ft_model.get_dimension())  # Return zero vector

    return np.mean(word_vectors, axis=0)  # Apply mean pooling

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


def compute_contrastive_loss(vision_embeds, text_embeds,  labels, temperature=0.07, use_class_mask= True):
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
    
    if use_class_mask:
        # Positive pairs (vision embedding and corresponding text embedding of the same class)
        pos_pairs = torch.exp(similarity_matrix) * mask
        # print("use class mask+++++++++++++++++++++++")
    else:
        pos_pairs = torch.exp(similarity_matrix)
        # print("not using class mask-----------------")
    
    # Sum of exponential similarities (both positive and negative samples)
    exp_similarity_matrix = torch.exp(similarity_matrix)
    neg_pairs_sum = exp_similarity_matrix.sum(dim=-1) - pos_pairs.sum(dim=-1)  # Exclude the positives from the sum
    
    # Compute the contrastive loss using NT-Xent formula
    contrastive_loss = -torch.log((pos_pairs.sum(dim=-1)) / (pos_pairs.sum(dim=-1) + neg_pairs_sum))
    
    # Return the mean loss across the batch
    return contrastive_loss.mean()


def orthogonal_regularization(text_embeds, margin=1e-5):
    """
    Encourage text embeddings to be orthogonal (dissimilar) by minimizing 
    the absolute value of their pairwise dot products.
    """
    # Normalize the text embeddings
    text_embeds = F.normalize(text_embeds, dim=-1)

    # Compute pairwise dot product between all text embeddings
    similarity_matrix = torch.matmul(text_embeds, text_embeds.T)
    
    # Create a mask to ignore the diagonal (self-similarity)
    identity_mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device)
    
    # Compute the orthogonality loss (minimize off-diagonal similarity)
    orthogonality_loss = torch.abs(similarity_matrix * (1 - identity_mask)).mean()
    
    # Penalize the similarity if it's above a small margin (enforcing dissimilarity)
    orthogonality_loss = torch.clamp(orthogonality_loss - margin, min=0.0)
    
    return orthogonality_loss


@ALGORITHMS.register('textmatch')
class TextMatch(AlgorithmBase):
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
        self.apply_mixup = args.apply_mixup
        self.use_class_maskoncontrastive = args.use_class_maskoncontrastive
        print(f"-----Apply Mixup----------->{self.apply_mixup}")
        
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
        elif args.dataset == 'eurosat':
            from semilearn.datasets import eurosat_label_name, eurosat_label_text
            label_name = eurosat_label_text if self.long_text else eurosat_label_name
        else:
            raise Exception(f"Unknown label name for dataset {args.dataset}")

        label_texts = [f"{label}" for label in label_name.values()]
        print("================Using lable_name as the text supervision================")
        print(label_name)
        print("========================================================================")

        if self.text_model == "clip":
            # CLIP Model
            print("using clip as text encoder!")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            # label_name_inputs = self.clip_processor(text=label_texts, return_tensors="pt", padding=True)
            label_name_inputs = self.clip_processor(text=label_texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
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
        
        elif self.text_model == "sentence-transformer":
            # SentenceTransformers Model
            print("Using SentenceTransformers as text encoder!")
            # Load a lightweight text embedding model
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").cuda()
            with torch.no_grad():
                # Directly encode label_texts
                self.text_embeddings_bank = model.encode(label_texts, convert_to_tensor=True).cuda()  # Keep embeddings on GPU
        

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
        
        elif self.text_model == "fasttext":
            print("Using FastText as text encoder!")

            # Load the pre-trained FastText model
            import fasttext.util
            fasttext.util.download_model('en', if_exists='ignore')  
            ft_model = fasttext.load_model('cc.en.300.bin')

            # Convert label names into text
            label_texts = [f"{label}" for label in label_name.values()]

            with torch.no_grad():
                # Convert labels to embeddings using FastText + Mean Pooling
                self.text_embeddings_bank = torch.tensor(
                    np.array([sentence_to_vector(text, ft_model) for text in label_texts])
                ).cuda()

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


    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FreeMatchThresholdingHook(num_classes=self.num_classes, momentum=self.args.ema_p), "MaskingHook")
        super().set_hooks()


    def proj_text(self):
        return self.model.module.projector(text_embs=self.text_embeddings_bank)


    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]
        projected_text_embeds_bank = self.model.module.projector(text_embs=self.text_embeddings_bank)
        # print("projected_text_embeds_bank.size()", projected_text_embeds_bank.size())
        # Apply orthogonal regularization using pseudo labels
        ortho_loss = orthogonal_regularization(projected_text_embeds_bank)
        mixup_now = np.random.rand() > 0.5
        # Apply Mixup or CutMix        
        if self.apply_mixup and mixup_now:
            if np.random.rand() > 0.5:
                x_lb, y_a, y_b, lam = mixup_data(x_lb, y_lb, alpha=0.2)
            else:
                x_lb, y_a, y_b, lam = cutmix_data(x_lb, y_lb, alpha=0.2)

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

            
            # Apply mixup/cutmix to the supervised loss
            if self.apply_mixup and mixup_now:
                sup_loss = lam * self.ce_loss(logits_x_lb, y_a, reduction='mean') + (1 - lam) * self.ce_loss(logits_x_lb, y_b, reduction='mean')
            else:
                sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

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
            text_embeds_x_lb = projected_text_embeds_bank[y_lb]
            
            # Apply the mask to filter out the embeddings where mask == 1
            vision_embeds_ulb_s = feats_x_ulb_s[mask.bool()]
            vision_embeds_ulb_w = feats_x_ulb_w[mask.bool()]
            text_embeds_ulb_w = projected_text_embeds_bank[pseudo_label[mask.bool()]]
            
            labels = pseudo_label[mask.bool()]

            vision_embeds = torch.cat([vision_embeds_x_lb, vision_embeds_ulb_w, vision_embeds_ulb_s])
            text_embeds = torch.cat([text_embeds_x_lb, text_embeds_ulb_w, text_embeds_ulb_w])
            
            labels = torch.cat([y_lb, labels, labels])
            
            contrastive_loss= compute_contrastive_loss(vision_embeds=vision_embeds,
                                        text_embeds=text_embeds,
                                        labels=labels, 
                                        temperature=self.contras_tempeture,
                                        use_class_mask = self.use_class_maskoncontrastive)
            
            contrastive_loss_value = contrastive_loss.item()


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
            SSL_Argument('--long_text', str2bool, True),
            SSL_Argument('--apply_mixup', str2bool, False),
            SSL_Argument('--save_text_embs', str2bool, False),
            SSL_Argument('--use_class_maskoncontrastive', str2bool, True)
        ]