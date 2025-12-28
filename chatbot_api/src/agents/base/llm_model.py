from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig
import logging
import torch
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


def get_llm(model_name: str, max_new_tokens: int = 1024, **kwargs):
    use_cuda = torch.cuda.is_available()
    quant_cfg = nf4_config if use_cuda else None
    dtype = torch.bfloat16 if use_cuda else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        torch_dtype=dtype,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_pipeline = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        device_map="auto",
    )

    hf_pipeline = HuggingFacePipeline(pipeline=model_pipeline, **kwargs)
    return hf_pipeline

