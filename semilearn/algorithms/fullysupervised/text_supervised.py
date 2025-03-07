# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn.functional as F

from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS

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


@ALGORITHMS.register('textsupervised')
class TextSupervised(AlgorithmBase):
    """
        Train a fully supervised model using labeled data only. This serves as a baseline for comparison.

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
        """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        self.vt_proj = args.vt_proj
        self.text_model = args.text_model
        self.long_text = args.long_text
        self.init_text_embeding(args=args)
        

        super().__init__(args, net_builder, tb_log, logger)

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

    def train_step(self, x_lb, y_lb):
        projected_text_embeds_bank = self.model.module.projector(text_embs=self.text_embeddings_bank)
        ortho_loss = orthogonal_regularization(projected_text_embeds_bank)
    
        # inference and calculate sup/unsup losses
        with self.amp_cm():
            outs_x_lb = self.model(x_lb)
            logits_x_lb = outs_x_lb['logits']
            feats_x_lb = outs_x_lb['feat']

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

        vision_embeds = feats_x_lb
        text_embeds = projected_text_embeds_bank[y_lb]
        labels = y_lb

        contrastive_loss= compute_contrastive_loss(vision_embeds=vision_embeds,
                            text_embeds=text_embeds,
                            labels=labels, 
                            temperature=self.contras_tempeture,
                            use_class_mask = self.use_class_maskoncontrastive)

        total_loss = sup_loss  +  self.lambda_c * contrastive_loss + \
                        ortho_loss * self.lambda_ortho
        
        out_dict = self.process_out_dict(loss=total_loss)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         contrastive_loss = contrastive_loss,
                                         ortho_loss = ortho_loss.item(),
                                         total_loss=total_loss.item(),)
        return out_dict, log_dict

    
    def train(self):
        # lb: labeled, ulb: unlabeled
        self.model.train()
        self.call_hook("before_run")
            
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            
            # prevent the training iterations exceed args.num_train_iter
            if self.it > self.num_train_iter:
                break

            self.call_hook("before_train_epoch")

            for data_lb in self.loader_dict['train_lb']:

                # prevent the training iterations exceed args.num_train_iter
                if self.it > self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb))
                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")
        self.call_hook("after_run")
    

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--lambda_c', float, 0.1),
            SSL_Argument('--contras_tempeture', float, 0.07),
            SSL_Argument('--lambda_ortho', float, 0.01),
            SSL_Argument('--vt_proj', str, 'linear'),
            SSL_Argument('--text_model', str, 'clip'),
            SSL_Argument('--long_text', str2bool, True),
            SSL_Argument('--save_text_embs', str2bool, False),
            SSL_Argument('--use_class_maskoncontrastive', str2bool, True)
        ]


# ALGORITHMS['supervised'] = FullySupervised
