"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import os
from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM
# from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
from lavis.models.blip_models.blip_outputs import Blip2Output, ExtendedBlipOutput
from transformers import AutoTokenizer, OPTForCausalLM, OPTConfig, LlamaTokenizer,AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import transformers
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from peft.utils.save_and_load import set_peft_model_state_dict
import random
from torch.utils.checkpoint import checkpoint
from lavis.models.ci_modules.visual_ci import LocalSample, GlobalSample, VDM, CrossAttention

@registry.register_model("blip2_opt")
class Blip2OPT(Blip2Base):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b.yaml",
        "pretrain_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b.yaml",
        "caption_coco_opt2.7b": "configs/models/blip2/blip2_caption_opt2.7b.yaml",
        "caption_coco_opt6.7b": "configs/models/blip2/blip2_caption_opt6.7b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        opt_model="facebook/opt-2.7b",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.27"), "BLIP-2 OPT requires transformers>=4.27"
        
        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model, torch_dtype=torch.float16
        )
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)
        
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None       

    def forward(self, samples):
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

        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

        self.opt_tokenizer.padding_side = "right"

        text = [t + "\n" for t in samples["text_input"]]

        opt_tokens = self.opt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

        empty_targets = (
            torch.ones(atts_opt.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}

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

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            prompt = [prompt] * image.size(0)

            opt_tokens = self.opt_tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            
            # new version for transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt,inputs_embeds],dim=1)
            
            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds, 
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
                            
            # previous version for transformers<4.27
            # if use_nucleus_sampling:
            #     query_embeds = inputs_opt.repeat_interleave(num_captions, dim=0)
            #     num_beams = 1
            # else:
            #     query_embeds = inputs_opt.repeat_interleave(num_beams, dim=0)

            # outputs = self.opt_model.generate(
            #     input_ids=input_ids,
            #     query_embeds=query_embeds,
            #     attention_mask=attention_mask,
            #     do_sample=use_nucleus_sampling,
            #     top_p=top_p,
            #     temperature=temperature,
            #     num_beams=num_beams,
            #     max_new_tokens=max_length,
            #     min_length=min_length,
            #     eos_token_id=self.eos_token_id,
            #     repetition_penalty=repetition_penalty,
            #     length_penalty=length_penalty,
            #     num_return_sequences=num_captions,
            # )

            # prompt_length = opt_tokens.input_ids.shape[1]
            # output_text = self.opt_tokenizer.batch_decode(
            #     outputs[:, prompt_length:], skip_special_tokens=True
            # )
            
            output_text = [text.strip() for text in output_text]
            return output_text
        
        
    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=0,
        **kwargs
    ):
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

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if isinstance(samples["text_input"], str):
                samples["text_input"] = [samples["text_input"]]
            if prompt:
                text_input = [prompt.format(question) for question in samples["text_input"]]
            else:
                text_input = samples["text_input"]

            self.opt_tokenizer.padding_side = "left"
            opt_tokens = self.opt_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)
        
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            
            # require transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt,inputs_embeds],dim=1)
            
            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                eos_token_id=self.eos_token_id,
                length_penalty=length_penalty,
            )
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
        if self._apply_lemmatizer or ("apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]):
            output_text = self._lemmatize(output_text)

        return output_text
    
    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer
        
    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        opt_model = cfg.get("opt_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )
        model.load_checkpoint_from_config(cfg)

        return model

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[]):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False
        
@registry.register_model("mini_gpt4_prompt")
class MiniGPT4_Prompt(Blip2Base):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/minigpt4.yaml",
        "pretrain_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b.yaml",
        "pretrain_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b.yaml",
        "caption_coco_opt2.7b": "configs/models/blip2/blip2_caption_opt2.7b.yaml",
        "caption_coco_opt6.7b": "configs/models/blip2/blip2_caption_opt6.7b.yaml",
    }

    def __init__(
        self,
        vit_model="medical_mae_vit_b",
        vit_ckp_path = "C:/Users/lguo5/.cache/torch/hub/checkpoints/vit-b_CXR_0.5M_mae.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        pretrained_Qformer=None,
        num_query_token=64,
        # opt_model="facebook/opt-2.7b",
        llama_model="",
        prompt_path="",
        prompt_template="",
        refinement_prompt_path="",
        end_sym='\n',
        # prompt="",
        max_txt_len=60,
        apply_lemmatizer=False,
        
        # configurations of the pretraining stage
        freeze_llama=False,
        pretraining_ckpt=None,
        low_resource=False,  # use 8 bit and put vit in cpu
        # lora configurations
        use_lora=True,
        lora_rank=32,
        lora_alpha=32,
        lora_dropout=0.1,
        bias = "none", 
        target_modules = ["q_proj",
        "v_proj", 
        "k_proj",
        "o_proj"],
        # Q-former generates coarse text
        use_nucleus_sampling=False,
        num_beams=5,
        min_text_len=10,
        top_p=0.9,
        
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.43.0"), "Llama 3.2 requires transformers>=4.43.0"
        
        self.tokenizer = self.init_tokenizer()
        self.use_lora = use_lora
        peft_config = LoraConfig(inference_mode=False, r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=bias, target_modules = target_modules)
        
        # use medical mae vit as visiual encoder
        if vit_ckp_path is not None:
            self.visual_encoder, self.ln_vision = self.init_medical_vision_encoder(
                vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision, vit_ckp_path
            )
        # self.visual_encoder, self.ln_vision = self.init_vision_encoder(
        #     vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        # )
        else:
            self.visual_encoder, self.ln_vision = self.init_vision_encoder(
                vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
            )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        # Q-former
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])
        # if pretrained_Qformer:
        #     logging.info(f"Loading custom Q-former weights from {pretrained_Qformer}")
        #     custom_ckpt = torch.load(pretrained_Qformer, map_location="cpu")
        #     custom_state_dict = custom_ckpt["model"] if "model" in custom_ckpt else custom_ckpt
            
        #     msg = self.Qformer.load_state_dict(custom_state_dict, strict=False)
        #     logging.info(f"Q-former loading message: {msg}")
        # self.Qformer.cls = None
        # self.Qformer.bert.embeddings.word_embeddings = None
        # self.Qformer.bert.embeddings.position_embeddings = None
        # for layer in self.Qformer.bert.encoder.layer:
        #     layer.output = None
        #     layer.intermediate = None
        # freeze Q-former
        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")
        else:
            logging.info("train Qformer")
        self.use_nucleus_sampling=use_nucleus_sampling
        self.num_beams=num_beams
        self.max_length=max_txt_len
        self.min_length=min_text_len
        self.top_p=top_p
        
        # LLAMA
        print(f"\n=== Initial GPU Memory ===")
        print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        print(f"Reserved: {torch.cuda.memory_reserved()/1024**2:.1f}MB")
        logging.info("Loading LLAMA")
        self.low_resource = low_resource
        # self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        # self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        print(f"\n=== GPU Memory Before Loading LLaMA ===")
        print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        
        # if torch.distributed.is_initialized():
        #     local_rank = torch.distributed.get_rank()
        # else:
        #     local_rank = int(os.environ.get('LOCAL_RANK', 0))

        self.llama_tokenizer = AutoTokenizer.from_pretrained(
                llama_model,
        )
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_model = AutoModelForCausalLM.from_pretrained(
            llama_model,
            torch_dtype=torch.float16,
            pad_token_id=self.llama_tokenizer.pad_token_id
            
        )
        print(f"eos_token: {self.llama_tokenizer.eos_token}")  # <|eot_id|>
        print(f"eos_token_id: {self.llama_tokenizer.eos_token_id}")
        print(f"\n=== GPU Memory After Loading LLaMA ===")
        print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        print(f"Reserved: {torch.cuda.memory_reserved()/1024**2:.1f}MB")
        # if not self.low_resource:
        #     max_memory = {
        #         0: "8GB",       # GPU
        #         "cpu": "32GB"   # CPU
        #     }
        #     self.llama_model = LlamaForCausalLM.from_pretrained(
        #             llama_model,
        #             # load_in_8bit=True,
        #             # device_map={'': local_rank}
        #             # device_map="auto",
        #             torch_dtype=torch.float16,
        #             # max_memory=max_memory
        #         )
        #     torch.cuda.empty_cache()
        #     print(f"\n=== GPU Memory After Loading LLaMA ===")
        #     print("local_rank: ", local_rank)
        #     print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        # else:
        #     max_memory = {
        #         0: "8GB",       # GPU
        #         "cpu": "32GB"   # CPU
        #     }
        #     self.llama_model = LlamaForCausalLM.from_pretrained(
        #             llama_model,
        #             # torch_dtype=torch.float16,
        #             device_map="auto",  # Automatically manage device placement
        #             load_in_8bit=True,  # Load in 8-bit precision
        #             low_cpu_mem_usage=True,
        #             max_memory=max_memory
        #         )
        # self.llama_model.gradient_checkpointing_enable()
        
        if freeze_llama:
            print("Freeze LLaMA model.")
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
        else:
            if self.use_lora:
                print("Freeze LLaMA model still, using LoRA to update.")
                for name, param in self.llama_model.named_parameters():
                    param.requires_grad = False
                self.peft_model = get_peft_model(self.llama_model, peft_config)
                # self.peft_model.gradient_checkpointing_enable()
                # for module in self.peft_model.modules():
                #     if hasattr(module, 'gradient_checkpointing'):
                #         module.gradient_checkpointing = True
                torch.cuda.empty_cache()
                # print("LoRA model gradient checkpointing enabled")
                print(f"\n=== GPU Memory After using LoRA ===")
                print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
            else:
                print("Do full parameters tuning.")
        if pretraining_ckpt is not None:
            if self.use_lora:
                print("LLaMA LoRA model loaded.")
                full_state_dict = torch.load(pretraining_ckpt, map_location='cpu')['module']
                set_peft_model_state_dict(self.peft_model, full_state_dict)    
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
         
        # OPT
        # self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        # self.opt_model = OPTForCausalLM.from_pretrained(
        #     opt_model, torch_dtype=torch.float16
        # )
        # for name, param in self.opt_model.named_parameters():
        #     param.requires_grad = False
        # self.eos_token_id = self.opt_tokenizer(
        #     "\n", add_special_tokens=False
        # ).input_ids[0]

        # self.opt_proj = nn.Linear(
        #     self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        # )
        self.end_sym = self.llama_tokenizer.eos_token
        self.max_txt_len = max_txt_len
        # self.prompt = prompt
        # prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        # self.prompt_length = prompt_tokens.attention_mask.sum(1)
        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            if not any("###Human:" in p for p in filted_prompts):
                self.refinement_prompt_list = [
                    prompt_template.format(p) for p in filted_prompts
                ]
            else:
                self.refinement_prompt_list = filted_prompts
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []
        # load refinement prompt  
        if refinement_prompt_path:
            with open(refinement_prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            if not any("###Human:" in p for p in filted_prompts):
                self.refinement_prompt_list = [
                    prompt_template.format(p) for p in filted_prompts
                ]
            else:
                self.refinement_prompt_list = filted_prompts
            print('Load {} refinement prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.refinement_prompt_list)))
            
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None   
        self.stop_words = [self.llama_tokenizer.eos_token]
    
        
    # def prompt_wrap(self, img_embeds, atts_img, prompt):
    #     if prompt:
    #         batch_size = img_embeds.shape[0]
    #         p_before, p_after = prompt.split('<ImageHere>')
    #         p_before_tokens = self.llama_tokenizer(
    #             p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
    #         p_after_tokens = self.llama_tokenizer(
    #             p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
    #         p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
    #         p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
    #         wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
    #         wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
    #         print("Final wrapped embeddings shape:", wrapped_img_embeds.shape)
    #         print("Final attention mask shape:", wrapped_atts_img.shape)
    #         return wrapped_img_embeds, wrapped_atts_img
    #     else:
    #         return img_embeds, atts_img
    def print_gpu_memory(self, message=""):
        print(f"\n=== GPU Memory ({message}) ===")
        print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        print(f"Cached: {torch.cuda.memory_reserved()/1024**2:.1f}MB")
        print(f"Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1024**2:.1f}MB")
        
    def prompt_wrap(self, img_embeds, atts_img, prompts):
        """
        Process prompts in batch
        Args:
            img_embeds: Image embeddings [batch_size, seq_len, hidden_dim]
            atts_img: Attention mask [batch_size, seq_len] 
            prompts: List of prompts with length equal to batch_size
        """
        
        if prompts:
            
            batch_size = img_embeds.shape[0]
            device = img_embeds.device
            # print(f"Image embeddings device: {device}")
            # print(f"Original embed_tokens device: {next(self.llama_model.model.embed_tokens.parameters()).device}")
            # Check if number of prompts matches batch size
            assert len(prompts) == batch_size, f"Number of prompts {len(prompts)} must equal batch size {batch_size}"
            
            # Store embeddings for each sample
            all_embeddings = []
            max_len = 0
            embed_tokens = self.llama_model.model.embed_tokens.to(device)
            # print(f"New embed_tokens device: {next(embed_tokens.parameters()).device}")
            # First process all prompts and find max length
            processed_prompts = []
            for prompt in prompts:
                p_before, p_after = prompt.split('<ImageHere>')
                
                # p_before_tokens = self.llama_tokenizer(
                #     p_before, return_tensors="pt", add_special_tokens=False
                # ).to(img_embeds.device)
                # p_after_tokens = self.llama_tokenizer(
                #     p_after, return_tensors="pt", add_special_tokens=False
                # ).to(img_embeds.device)
                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False
                )
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False
                )
                p_before_tokens = p_before_tokens.input_ids.to(device)
                p_after_tokens = p_after_tokens.input_ids.to(device)
                # print(f"Token devices - before: {p_before_tokens.device}, after: {p_after_tokens.device}")
            
                processed_prompts.append({
                    'before': p_before_tokens,
                    'after': p_after_tokens
                })
                
                # Calculate total length for this sample
                current_len = (p_before_tokens.shape[1] + 
                                img_embeds.shape[1] + 
                                p_after_tokens.shape[1])
                max_len = max(max_len, current_len)
            
            # Build embeddings for each sample
            for i in range(batch_size):
                # print(f"Processing batch {i}, device: {device}")
                # Get embeddings for prefix and suffix
                # p_before_embeds = self.llama_model.model.embed_tokens(
                #     processed_prompts[i]['before'].input_ids
                # )
                # p_after_embeds = self.llama_model.model.embed_tokens(
                #     processed_prompts[i]['after'].input_ids
                # )
                
                # p_before_embeds = embed_tokens(
                #     processed_prompts[i]['before']['input_ids']
                # )
                # p_after_embeds = embed_tokens(
                #     processed_prompts[i]['after']['input_ids']
                # )
                
                p_before_embeds = embed_tokens(processed_prompts[i]['before'])
                p_after_embeds = embed_tokens(processed_prompts[i]['after'])
                # print(f"Embedding devices - before: {p_before_embeds.device}, after: {p_after_embeds.device}")
                
                # Concatenate prefix, image and suffix embeddings
                sample_embeds = torch.cat([
                    p_before_embeds, 
                    img_embeds[i:i+1], 
                    p_after_embeds
                ], dim=1)
                
                # Pad to max length if needed
                if sample_embeds.shape[1] < max_len:
                    pad_length = max_len - sample_embeds.shape[1]
                    padding = torch.zeros(
                        1, pad_length, sample_embeds.shape[-1], 
                        dtype=sample_embeds.dtype, 
                        device=device
                    )
                    sample_embeds = torch.cat([sample_embeds, padding], dim=1)
                
                all_embeddings.append(sample_embeds)
            
            # Stack all sample embeddings
            wrapped_img_embeds = torch.cat(all_embeddings, dim=0)
            # Expand attention mask to match sequence length
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            # print(f"Final embeddings device: {wrapped_img_embeds.device}")  
            # print(f"Final attention mask device: {wrapped_atts_img.device}")  
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img
    
    
    def forward(self, samples):
        torch.cuda.empty_cache()
        # self.print_gpu_memory("Start of forward")
        with torch.no_grad():
        # visual encoder
        
            with self.maybe_autocast():
                image_features, _ = self.visual_encoder(image)
                image_embeds = self.ln_vision(image_features)
                
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image_embeds.device
            )
                
            # use Q-former generate coarse text
            model_kwargs = {
                "encoder_hidden_states": image_embeds,
                "encoder_attention_mask": image_atts,
            }
            input_ids = (
                torch.LongTensor(image.size(0), 1)
                .fill_(self.tokenizer.bos_token_id)
                .to(image_embeds.device)
            )
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            
            coarse_outputs = self.Qformer.generate(
                input_ids=input_ids,
                query_embeds=query_tokens,
                max_length=self.max_length,
                min_length=self.min_length,
                num_beams=self.num_beams,
                do_sample=self.use_nucleus_sampling,
                top_p=self.top_p,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                **model_kwargs
            )
            coarse_text = self.tokenizer.batch_decode(coarse_outputs, skip_special_tokens=True)
            # print("coarse text: ", coarse_text)
            prompts = []
            for text in coarse_text:
                before_report, after_report = random.choice(self.refinement_prompt_list).split("<ReportHere>")
                prompt = before_report + text + after_report

                prompts.append(prompt)
            
            # before_report, after_report = random.choice(self.refinement_prompt_list).split("<ReportHere>")
            # prompt = before_report + coarse_text + after_report
            # print("prompts: ",prompt)
            # use Q-former to extract image embeds
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
                
            
        inputs_img_llama = self.llama_proj(query_output.last_hidden_state).to(image_embeds.device)
        
        atts_img_llama = torch.ones(inputs_img_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
        
        inputs_img_llama, atts_img_llama = self.prompt_wrap(inputs_img_llama, atts_img_llama, prompts)
        # print("inputs_img_llama: ", inputs_img_llama)
        self.llama_tokenizer.padding_side = "right"
        
        # preprocess
        gt_report = [t + self.end_sym for t in samples["text_input"]]
        # print("gt_report: ", gt_report)
        gt_report_tokens = self.llama_tokenizer(
            gt_report,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image_embeds.device)
        
        # Create true labels for the gt report part， -100 for pad tokens
        gt_report_targets = gt_report_tokens.input_ids.masked_fill(
            gt_report_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        # Create -100 labels for the image and coarse text parts
        gt_empty_targets = (
            torch.ones([atts_img_llama.shape[0], atts_img_llama.shape[1] + 1],
                    dtype=torch.long).to(image_embeds.device).fill_(-100)  # plus one for bos
        )
        gt_report_targets = torch.cat([gt_empty_targets, gt_report_targets], dim=1)
        batch_size = inputs_img_llama.shape[0]
        bos = torch.ones([batch_size, 1],
                        dtype=gt_report_tokens.input_ids.dtype,
                        device=gt_report_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.peft_model.model.model.embed_tokens(bos)
        atts_bos = atts_img_llama[:, :1]
        gt_report_embeds = self.peft_model.model.model.embed_tokens(gt_report_tokens.input_ids)
        gt_inputs_embeds = torch.cat([bos_embeds, inputs_img_llama, gt_report_embeds], dim=1)
        gt_attention_mask = torch.cat([atts_bos, atts_img_llama, gt_report_tokens.attention_mask], dim=1)

        def check_tensor(tensor, name):
            if torch.isnan(tensor).any():
                print(f"{name} contains NaN!")
            if torch.isinf(tensor).any():
                print(f"{name} contains Inf!")
        check_tensor(gt_inputs_embeds, "inputs_embeds")
        check_tensor(gt_report_targets, "labels")
        # self.print_gpu_memory("Before PEFT model forward")
        # forwarding
        with self.maybe_autocast():
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            outputs = self.peft_model(
                inputs_embeds=gt_inputs_embeds,
                attention_mask=gt_attention_mask,
                return_dict=True,
                labels=gt_report_targets,
            )
            # if hasattr(outputs, 'logits'):
            #     print("Logits stats:",
            #         "mean:", outputs.logits.mean().item(),
            #         "std:", outputs.logits.std().item(),
            #         "max:", outputs.logits.max().item(),
            #         "min:", outputs.logits.min().item())
        # if torch.isnan(outputs.loss):
        #     print("Loss is NaN!")
            # print("Labels distribution:", torch.unique(gt_report_targets, return_counts=True))
            # print("Attention mask sum:", gt_attention_mask.sum().item())
             # 检查是否所有label都是-100
            # valid_labels = (gt_report_targets != -100).sum().item()
            # print("Number of valid labels:", valid_labels)
        loss = outputs.loss
        
        return {"loss": loss}
        
            
            
            
        # inputs_opt = self.opt_proj(query_output.last_hidden_state)
        # atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

        # self.opt_tokenizer.padding_side = "right"

        # text = [t + "\n" for t in samples["text_input"]]

        # opt_tokens = self.opt_tokenizer(
        #     text,
        #     return_tensors="pt",
        #     padding="longest",
        #     truncation=True,
        #     max_length=self.max_txt_len,
        # ).to(image.device)

        # targets = opt_tokens.input_ids.masked_fill(
        #     opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        # )
        # if self.prompt:
        #     targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

        # empty_targets = (
        #     torch.ones(atts_opt.size(), dtype=torch.long).to(image.device).fill_(-100)
        # )
        # targets = torch.cat([empty_targets, targets], dim=1)

        # inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        # inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        # attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        # with self.maybe_autocast():
        #     outputs = self.opt_model(
        #         inputs_embeds=inputs_embeds,
        #         attention_mask=attention_mask,
        #         return_dict=True,
        #         labels=targets,
        #     )
        # loss = outputs.loss

        # return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        # num_beams=5,
        # max_length=30,
        # min_length=1,
        # top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
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
        gt_text = samples["text_input"]
        with self.maybe_autocast():
            image_features, _ = self.visual_encoder(image)
            image_embeds = self.ln_vision(image_features)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image_embeds.device
            )

            # use Q-former generate coarse text
            model_kwargs = {
                "encoder_hidden_states": image_embeds,
                "encoder_attention_mask": image_atts,
            }
            input_ids = (
                torch.LongTensor(image.size(0), 1)
                .fill_(self.tokenizer.bos_token_id)
                .to(image_embeds.device)
            )
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            
            coarse_outputs = self.Qformer.generate(
                input_ids=input_ids,
                query_embeds=query_tokens,
                max_length=self.max_length,
                min_length=self.min_length,
                num_beams=self.num_beams,
                do_sample=self.use_nucleus_sampling,
                top_p=self.top_p,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                **model_kwargs
            )
            coarse_text = self.tokenizer.batch_decode(coarse_outputs, skip_special_tokens=True)
            prompts = []
            for text in coarse_text:
                before_report, after_report = random.choice(self.refinement_prompt_list).split("<ReportHere>")
                prompt = before_report + text + after_report
                prompts.append(prompt)

            # before_report, after_report = random.choice(self.refinement_prompt_list).split("<ReportHere>")
            # prompt = before_report + coarse_text + after_report
            
            # use Q-former to extract image embeds
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            inputs_img_llama = self.llama_proj(query_output.last_hidden_state)
            atts_img_llama = torch.ones(inputs_img_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
            inputs_img_llama, atts_img_llama = self.prompt_wrap(inputs_img_llama, atts_img_llama, prompts)
            
            batch_size = inputs_img_llama.shape[0]
            bos = torch.ones([batch_size, 1],
                            dtype=torch.long,
                            device=image_embeds.device) * self.llama_tokenizer.bos_token_id
            bos_embeds = self.peft_model.model.model.embed_tokens(bos)
            atts_bos = atts_img_llama[:, :1]
            
            inputs_embeds = torch.cat([bos_embeds, inputs_img_llama], dim=1)
            attention_mask = torch.cat([atts_bos, atts_img_llama], dim=1)
            
            # stop_words_ids = [torch.tensor(self.llama_tokenizer(self.stop_words, add_special_tokens=False)['input_ids'], dtype=torch.long, device='cuda') for word in self.stop_words]
            stop_words_ids = [torch.tensor([self.llama_tokenizer.eos_token_id], device=image_embeds.device)]
            # stop_words_ids = [torch.tensor([835]).to(device), torch.tensor([2277, 29937]).to(device)]  # '###' can be encoded in two different ways.
            stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
            input_length = inputs_embeds.shape[1]
            max_length = input_length + self.max_txt_len
            # print("max_length: ", max_length)
            with self.maybe_autocast():
                outputs = self.peft_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    # max_new_tokens=self.max_txt_len,
                    max_new_tokens=max_length,
                    # max_length = max_length,
                    stopping_criteria=stopping_criteria,
                    num_beams=self.num_beams,              
                    min_length=self.min_length,
                    # length_penalty=1.0,       
                    do_sample=False,         
                    repetition_penalty=1.0,   
                    temperature=temperature,
                    pad_token_id=self.llama_tokenizer.pad_token_id,
                    eos_token_id=self.llama_tokenizer.eos_token_id,
                    output_hidden_states=False,  
                    output_scores=False,         
                    return_dict_in_generate=True 
                )
            # print("model_max_length:", getattr(self.peft_model.config, 'max_length', 'Not set'))
            # Only the generated new token is taken, excluding the input token
            output_tokens = outputs.sequences[:, inputs_embeds.shape[1]:]
            # sequences = outputs.sequences
            # print("outputs: ", sequences.shape)
            # print("inputs_embeds: ", inputs_embeds.shape)
            # print("output_tokens: ", output_tokens.shape)
            output_text = self.llama_tokenizer.batch_decode(
                output_tokens, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            # print("output_text: ", output_text)
            return ExtendedBlipOutput(
            Q_former_predicted_captions=coarse_text,
            LLM_predicted_captions=output_text,
            gt_captions=gt_text
        )

            
            
            
            
            
            
            
            
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            prompt = [prompt] * image.size(0)

            opt_tokens = self.opt_tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            
            # new version for transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt,inputs_embeds],dim=1)
            
            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds, 
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
                            
            # previous version for transformers<4.27
            # if use_nucleus_sampling:
            #     query_embeds = inputs_opt.repeat_interleave(num_captions, dim=0)
            #     num_beams = 1
            # else:
            #     query_embeds = inputs_opt.repeat_interleave(num_beams, dim=0)

            # outputs = self.opt_model.generate(
            #     input_ids=input_ids,
            #     query_embeds=query_embeds,
            #     attention_mask=attention_mask,
            #     do_sample=use_nucleus_sampling,
            #     top_p=top_p,
            #     temperature=temperature,
            #     num_beams=num_beams,
            #     max_new_tokens=max_length,
            #     min_length=min_length,
            #     eos_token_id=self.eos_token_id,
            #     repetition_penalty=repetition_penalty,
            #     length_penalty=length_penalty,
            #     num_return_sequences=num_captions,
            # )

            # prompt_length = opt_tokens.input_ids.shape[1]
            # output_text = self.opt_tokenizer.batch_decode(
            #     outputs[:, prompt_length:], skip_special_tokens=True
            # )
            
            output_text = [text.strip() for text in output_text]
            return output_text
        
        
    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=0,
        **kwargs
    ):
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

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if isinstance(samples["text_input"], str):
                samples["text_input"] = [samples["text_input"]]
            if prompt:
                text_input = [prompt.format(question) for question in samples["text_input"]]
            else:
                text_input = samples["text_input"]

            self.opt_tokenizer.padding_side = "left"
            opt_tokens = self.opt_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)
        
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            
            # require transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt,inputs_embeds],dim=1)
            
            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                eos_token_id=self.eos_token_id,
                length_penalty=length_penalty,
            )
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
        if self._apply_lemmatizer or ("apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]):
            output_text = self._lemmatize(output_text)

        return output_text
    
    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer
        
    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "medical_mae_vit_b")
        vit_ckp_path = cfg.get("vit_ckp_path", None)
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        
        freeze_llama = cfg.get("freeze_llama", False)
        low_resource = cfg.get("low_resource", False)
        llama_model = cfg.get("llama_model", "Llama-3.2-1B-Instruct")
        
        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        pretrained_Qformer=cfg.get("pretrained_Qformer", None)
        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 60)
        min_text_len = cfg.get("min_len", 10)
        num_beams = cfg.get("num_beams", 5)
        end_sym = cfg.get("end_sym", '\n')
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        
        refinement_prompt_path = cfg.get("refinement_prompt_path", None)
        
        # some choices on PEFT methods
        use_lora = cfg.get("use_lora", True)
        lora_rank = cfg.get("lora_rank", None)
        lora_alpha = cfg.get("lora_alpha", None)
        lora_dropout = cfg.get("lora_dropout", None)
        target_modules= cfg.get("target_modules", None)
        
        model = cls(
            vit_model=vit_model,
            vit_ckp_path =vit_ckp_path,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            pretrained_Qformer=pretrained_Qformer,
            
            freeze_llama=freeze_llama,
            low_resource = low_resource,
            llama_model=llama_model,
            
            num_query_token=num_query_token,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            min_text_len=min_text_len,
            num_beams=num_beams,
            end_sym=end_sym,
            apply_lemmatizer=apply_lemmatizer,
            
            refinement_prompt_path=refinement_prompt_path,
            
            # lora configurations
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules
        )
        model.load_checkpoint_from_config(cfg)

        return model


@registry.register_model("moultiview_blip2_opt")
class MultiViewBlip2OPT(Blip2Base):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b.yaml",
        "pretrain_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b.yaml",
        "caption_coco_opt2.7b": "configs/models/blip2/blip2_caption_opt2.7b.yaml",
        "caption_coco_opt6.7b": "configs/models/blip2/blip2_caption_opt6.7b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=64,
        opt_model="facebook/opt-2.7b",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.27"), "BLIP-2 OPT requires transformers>=4.27"
        
        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model, torch_dtype=torch.float16
        )
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)
        
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None   
        # self.max_txt_len = max_txt_len
        self.num_query_token = num_query_token
        # Freeze the ln_vision
        # for name, param in self.ln_vision.named_parameters():
        #     param.requires_grad = False
        # self.ln_vision.eval()  
        # logging.info("freeze ln_vision")

        # # Freeze the Qformer
        # for name, param in self.Qformer.named_parameters():
        #     param.requires_grad = False
        # self.Qformer.eval() 
        # logging.info("freeze Qformer")
        
        # Initialize view embeddings
        # self.view_embeddings = nn.Parameter(
        #     torch.ones(2) * 0.02 
        # )    

    def forward(self, samples):
        image = samples["image"]
        with self.maybe_autocast():
            # image_embeds = self.ln_vision(self.visual_encoder(image))
            # Process images based on input dimension
            if len(image.shape) == 5:  # Multi-view input: [B, V, C, H, W]
                B, V, C, H, W = image.shape
                
                # Process each view separately
                image_embeds_list = []
                for v in range(V):
                    current_view = image[:, v]  # [B, C, H, W]
                    view_embeds = self.ln_vision(
                        self.visual_encoder(current_view)
                    )  # [B, num_patches+1, hidden_size]
                    
                    # Add view embedding
                    # print(f"view_embeds: {view_embeds.shape}")
                    # print(f"self.view_embeddings[v]: {self.view_embeddings}")
                    view_embeds = view_embeds + self.view_embeddings[v]
                    # print(f'after view_embeds: {view_embeds.shape}')
                    image_embeds_list.append(view_embeds)
                
                # Concatenate features from all views
                image_embeds = torch.cat(image_embeds_list, dim=1)  # [B, V*(num_patches+1), hidden_size]
                # print(f"image_embeds: {image_embeds.shape}")
            else:  # Single-view input: [B, C, H, W]
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

        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

        self.opt_tokenizer.padding_side = "right"

        text = [t + "\n" for t in samples["text_input"]]
        # i added 
        # if self.prompt:
        #     text = [self.prompt + t for t in text]
        opt_tokens = self.opt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

        empty_targets = (
            torch.ones(atts_opt.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}

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
            # image_embeds = self.ln_vision(self.visual_encoder(image))
            # Process images based on input dimension
            if len(image.shape) == 5:  # Multi-view input: [B, V, C, H, W]
                B, V, C, H, W = image.shape
                
                # Process each view separately
                image_embeds_list = []
                for v in range(V):
                    current_view = image[:, v]  # [B, C, H, W]
                    view_embeds = self.ln_vision(
                        self.visual_encoder(current_view)
                    )  # [B, num_patches+1, hidden_size]
                    
                    # Add view embedding
                    # print(f"view_embeds: {view_embeds.shape}")
                    # print(f"self.view_embeddings[v]: {self.view_embeddings}")
                    view_embeds = view_embeds + self.view_embeddings[v]
                    # print(f'after view_embeds: {view_embeds.shape}')
                    image_embeds_list.append(view_embeds)
                
                # Concatenate features from all views
                image_embeds = torch.cat(image_embeds_list, dim=1)  # [B, V*(num_patches+1), hidden_size]
                # print(f"image_embeds: {image_embeds.shape}")
            else:  # Single-view input: [B, C, H, W]
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

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt
            prompt = [prompt] * image.size(0)

            opt_tokens = self.opt_tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            
            # new version for transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt,inputs_embeds],dim=1)
            
            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds, 
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
                            
            # previous version for transformers<4.27
            # if use_nucleus_sampling:
            #     query_embeds = inputs_opt.repeat_interleave(num_captions, dim=0)
            #     num_beams = 1
            # else:
            #     query_embeds = inputs_opt.repeat_interleave(num_beams, dim=0)

            # outputs = self.opt_model.generate(
            #     input_ids=input_ids,
            #     query_embeds=query_embeds,
            #     attention_mask=attention_mask,
            #     do_sample=use_nucleus_sampling,
            #     top_p=top_p,
            #     temperature=temperature,
            #     num_beams=num_beams,
            #     max_new_tokens=max_length,
            #     min_length=min_length,
            #     eos_token_id=self.eos_token_id,
            #     repetition_penalty=repetition_penalty,
            #     length_penalty=length_penalty,
            #     num_return_sequences=num_captions,
            # )

            # prompt_length = opt_tokens.input_ids.shape[1]
            # output_text = self.opt_tokenizer.batch_decode(
            #     outputs[:, prompt_length:], skip_special_tokens=True
            # )
            
            output_text = [text.strip() for text in output_text]
            return output_text
        
        
    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=0,
        **kwargs
    ):
        image = samples["image"]
        with self.maybe_autocast():
            # image_embeds = self.ln_vision(self.visual_encoder(image))
            if len(image.shape) == 5:  # Multi-view input: [B, V, C, H, W]
                B, V, C, H, W = image.shape
                
                # Process each view separately
                image_embeds_list = []
                for v in range(V):
                    current_view = image[:, v]  # [B, C, H, W]
                    view_embeds = self.ln_vision(
                        self.visual_encoder(current_view)
                    )  # [B, num_patches+1, hidden_size]
                    
                    # Add view embedding
                    # print(f"view_embeds: {view_embeds.shape}")
                    # print(f"self.view_embeddings[v]: {self.view_embeddings}")
                    view_embeds = view_embeds + self.view_embeddings[v]
                    # print(f'after view_embeds: {view_embeds.shape}')
                    image_embeds_list.append(view_embeds)
                
                # Concatenate features from all views
                image_embeds = torch.cat(image_embeds_list, dim=1)  # [B, V*(num_patches+1), hidden_size]
                # print(f"image_embeds: {image_embeds.shape}")
            else:  # Single-view input: [B, C, H, W]
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

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if isinstance(samples["text_input"], str):
                samples["text_input"] = [samples["text_input"]]
            if prompt:
                text_input = [prompt.format(question) for question in samples["text_input"]]
            else:
                text_input = samples["text_input"]

            self.opt_tokenizer.padding_side = "left"
            opt_tokens = self.opt_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)
        
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            
            # require transformers>=4.27
            inputs_embeds = self.opt_model.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt,inputs_embeds],dim=1)
            
            outputs = self.opt_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                eos_token_id=self.eos_token_id,
                length_penalty=length_penalty,
            )
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
        if self._apply_lemmatizer or ("apply_lemmatizer" in samples.keys() and samples["apply_lemmatizer"]):
            output_text = self._lemmatize(output_text)

        return output_text
    
    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer
        
    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        opt_model = cfg.get("opt_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )
        model.load_checkpoint_from_config(cfg)

        return model
