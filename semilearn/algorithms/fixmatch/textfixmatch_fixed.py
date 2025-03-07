# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool

from transformers import CLIPModel, CLIPProcessor, BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel, \
      T5Tokenizer, T5EncoderModel, DebertaV2Tokenizer, DebertaV2Model


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


@ALGORITHMS.register('textfixmatchfixed')
class TextFixMatchFixed(AlgorithmBase):

    """
        FixMatch algorithm (https://arxiv.org/abs/2001.07685).

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
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        
        self.vt_proj = args.vt_proj
        self.text_model = args.text_model
        self.long_text = args.long_text
        self.init_text_embeding(args=args)

        super().__init__(args, net_builder, tb_log, logger) 
        # fixmatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label)
        self.threshold = args.p_threshold
        self.lambda_c = args.lambda_c
        self.lambda_ortho = args.lambda_ortho
        self.contras_tempeture = args.contras_tempeture
        self.use_class_maskoncontrastive = args.use_class_maskoncontrastive
        

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

    def init(self, T, p_cutoff, hard_label=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label

        self.copied_model = None  # Placeholder for the copied model
        self.threshold = 0.327  # Accuracy threshold for copying the model
        
        self.reached_threshold = False  # Flag to indicate if the threshold is reached
    
    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    

    def copy_model(self):
        import copy
        """Create a copy of the current model for pseudo-labeling."""
        self.copied_model = copy.deepcopy(self.model)
        self.copied_model.eval()  # Ensure the copied model is in evaluation mode

    def check_and_copy_model(self, current_accuracy):
        """Check if the accuracy threshold is reached and copy the model."""
        if current_accuracy >= self.threshold and not self.reached_threshold:
            print(f"Accuracy threshold {self.threshold * 100}% reached. Copying the model...")
            self.copy_model()
            self.reached_threshold = True


    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]
        projected_text_embeds_bank = self.model.module.projector(text_embs=self.text_embeddings_bank)
        # print("projected_text_embeds_bank.size()", projected_text_embeds_bank.size())
        # Apply orthogonal regularization using pseudo labels
        ortho_loss = orthogonal_regularization(projected_text_embeds_bank)
        

        # Check and copy the model if accuracy threshold is reached
        self.check_and_copy_model(self.best_eval_acc)

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
                    # Use the copied model for pseudo-labeling if threshold is reached
                    if self.reached_threshold:
                        pseudo_model = self.copied_model
                        self.use_c = 1.0
                    else:
                        self.use_c = 0.0
                        pseudo_model = self.model
                    outs_x_ulb_w = pseudo_model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            
            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            
            # if distribution alignment hook is registered, call it 
            # this is implemented for imbalanced algorithm - CReST
            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)

            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                               pseudo_label,
                                               'ce',
                                               mask=mask)

            # contrastive loss
            vision_embeds_x_lb = feats_x_lb
            text_embeds_x_lb = projected_text_embeds_bank[y_lb]
            
            # Apply the mask to filter out the embeddings where mask == 1
            vision_embeds_ulb_s = feats_x_ulb_s[mask.bool()]
            # vision_embeds_ulb_w = feats_x_ulb_w[mask.bool()]
            text_embeds_ulb_w = projected_text_embeds_bank[pseudo_label[mask.bool()]]
            
            labels = pseudo_label[mask.bool()]

            vision_embeds = torch.cat([vision_embeds_x_lb, vision_embeds_ulb_s])
            text_embeds = torch.cat([text_embeds_x_lb, text_embeds_ulb_w])
            
            labels = torch.cat([y_lb, labels])
            
            contrastive_loss= compute_contrastive_loss(vision_embeds=vision_embeds,
                                        text_embeds=text_embeds,
                                        labels=labels, 
                                        temperature=self.contras_tempeture,
                                        use_class_mask = self.use_class_maskoncontrastive)
            


            total_loss = sup_loss + self.lambda_u * unsup_loss + \
                        self.use_c * self.lambda_c * contrastive_loss + \
                        self.use_c * ortho_loss * self.lambda_ortho

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(),
                                         contrastive_loss = contrastive_loss.item(),
                                         ortho_loss = ortho_loss.item(),
                                         use_c = self.use_c,
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict
        

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--p_threshold', float, 0.30),
            
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
