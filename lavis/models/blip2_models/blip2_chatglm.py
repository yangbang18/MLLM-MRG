# Added by Yang B.
import logging
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.chatglm.tokenization_chatglm import ChatGLMTokenizer
from lavis.models.chatglm.modeling_chatglm import ChatGLMForConditionalGeneration

import transformers
transformers.logging.set_verbosity_error()


@registry.register_model("blip2_chatglm")
class Blip2ChatGLM(Blip2Base):
    """
    BLIP2 + ChatGLM model.
    Supported model types:
        - caption: pretrained model with ChatGLM model
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_chatglm", "caption")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "caption": "configs/models/blip2/blip2_caption_chatglm6b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        partial_freeze=0,
        num_query_token=32,
        chatglm_model="THUDM/chatglm-6b",
        input_prompt="Question: What is the radiology report for this image?\nAnswer:",
        max_txt_len=64,
        pre_seq_len=None,
        prefix_projection=False,
        vis_pre_seq_len=None,
        vit_path=None,
        freeze_q=False,
        cross_attention_freq=2,
        qformer_kwargs={},
        bert_model='bert-base-uncased',
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer(bert_model)

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision, vis_pre_seq_len, cached_file=vit_path
        )
        if vis_pre_seq_len is not None and vis_pre_seq_len > 0:
            logging.info("freeze vision encoder, but train soft prefix tokens")
        elif partial_freeze > 0:
            logging.info(f"train vision encoder staring from the {partial_freeze}-th layer")
            for n, p in self.visual_encoder.named_parameters():
                if 'block' in n:
                    num_block = n.split('.')[1]
                    if int(num_block) < partial_freeze:
                        p.requires_grad = False
                    else:
                        print('-', n)
                        p.requires_grad = True
                        p.data = p.data.float()
                else: 
                    p.requires_grad = False
        elif freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        logging.info(f"cross_attention_freq: {cross_attention_freq}")
        logging.info(f"qformer_kwargs: {qformer_kwargs}")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, 
            cross_attention_freq, 
            kwargs=qformer_kwargs,
            bert_model=bert_model,
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        if freeze_q:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            logging.info("freeze Q-former")

        self.chatglm_model = ChatGLMForConditionalGeneration.from_pretrained(
            chatglm_model, pre_seq_len=pre_seq_len, prefix_projection=prefix_projection,
            key_embed_dim=self.visual_encoder.num_features,
        )
        self.chatglm_tokenizer = ChatGLMTokenizer.from_pretrained(chatglm_model)

        if pre_seq_len is None or pre_seq_len == 0:
            # freeze ChatGLM manually
            for name, param in self.chatglm_model.named_parameters():
                param.requires_grad = False

        self.chatglm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.chatglm_model.config.hidden_size
        )

        self.num_query_token = num_query_token

        self.max_txt_len = max_txt_len
        self.input_prompt = input_prompt


    def get_prompts(self, samples, batch_size):
        if "input_prompt" in samples.keys():
            prompts = samples["input_prompt"]
            if isinstance(prompts, (list, tuple)):
                assert len(prompts) == batch_size
            else:
                prompts = [prompts] * batch_size
        else:
            prompts = [self.input_prompt] * batch_size
        return prompts

    def forward(self, samples):
        image = samples["image"]
        batch_size = image.size(0)

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        query_embeds = self.chatglm_proj(query_output.last_hidden_state)

        # tokenized.input_ids: (batch_size, seq_len)
        # tokenized.attention_mask: (batch_size, 1, seq_len, seq_len) # not used
        # tokenized.position_ids: (batch_size, 2, seq_len) # not used
        tokenized = self.chatglm_tokenizer(
            self.get_prompts(samples, batch_size),
            samples["text_input"],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        # construct labels
        assert self.chatglm_tokenizer.padding_side == 'left'
        labels = tokenized.input_ids.clone()
        for label in labels:
            bos_position = label.tolist().index(self.chatglm_tokenizer.bos_token_id)
            label[:bos_position] = -100
        
        prefix_labels = (
            torch.ones((batch_size, self.num_query_token), dtype=torch.long).to(image.device).fill_(-100)
        )
        labels = torch.cat([prefix_labels, labels], dim=1)

        with self.maybe_autocast():
            outputs = self.chatglm_model(
                input_ids=tokenized.input_ids,
                query_embeds=(query_embeds, image_embeds),
                return_dict=True,
                labels=labels,
            )
        lm_loss = outputs.loss
        loss = lm_loss

        out = {"loss": loss, "lm_loss": lm_loss}
        return out

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        do_not_activate_prefix_tokens: bool = False,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            query_embeds = self.chatglm_proj(query_output.last_hidden_state)

            input_prompt = self.get_prompts(samples, image.size(0))
            tokenized = self.chatglm_tokenizer(input_prompt, return_tensors="pt").to(
                image.device
            )

            if use_nucleus_sampling:
                query_embeds = query_embeds.repeat_interleave(num_captions, dim=0)
                num_beams = 1
            else:
                query_embeds = query_embeds.repeat_interleave(num_beams, dim=0)

            outputs = self.chatglm_model.generate(
                input_ids=tokenized.input_ids,
                query_embeds=(query_embeds, image_embeds),
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                do_not_activate_prefix_tokens=do_not_activate_prefix_tokens,
            )

            prompt_length = tokenized.input_ids.shape[1]
            output_text = self.chatglm_tokenizer.batch_decode(
                outputs[:, prompt_length:], skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
            return output_text

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        partial_freeze = cfg.get("partial_freeze", 0)

        chatglm_model = cfg.get("chatglm_model", "THUDM/chatglm-6b")
        input_prompt = cfg.get("input_prompt", "Question: What is the radiology report for this image?\nAnswer:")
        max_txt_len = cfg.get("max_txt_len", 64)
        pre_seq_len = cfg.get("pre_seq_len", None)
        prefix_projection = cfg.get("prefix_projection", False)

        vis_pre_seq_len = cfg.get("vis_pre_seq_len", None)
        vit_path = cfg.get("vit_path", None)
        freeze_q = cfg.get('freeze_q', False)

        cross_attention_freq = cfg.get('cross_attention_freq', 2)
        qformer_kwargs = {
            'num_hidden_layers': cfg.get('num_hidden_layers', 12),
            'random_init': cfg.get('random_init', False),
        }
        bert_model = cfg.get('bert_model', 'bert-base-uncased')

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            partial_freeze=partial_freeze,
            num_query_token=num_query_token,
            chatglm_model=chatglm_model,
            input_prompt=input_prompt,
            max_txt_len=max_txt_len,
            pre_seq_len=pre_seq_len,
            prefix_projection=prefix_projection,
            vis_pre_seq_len=vis_pre_seq_len,
            vit_path=vit_path,
            freeze_q=freeze_q,
            cross_attention_freq=cross_attention_freq,
            qformer_kwargs=qformer_kwargs,
            bert_model=bert_model,
        )

        model.load_checkpoint_from_config(cfg)

        return model
