from unsloth import FastVisionModel

from vlm_chartqa.config import LORA_RANK, MAX_SEQ_LEN, MODEL_NAME


def load_model(lora_path=None):
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=lora_path if lora_path else MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,  # False for LoRA 16bit
        fast_inference=False,  # Enable vLLM fast inference
        gpu_memory_utilization=0.8,  # Reduce if out of memory
    )

    if not lora_path:
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=LORA_RANK,
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
            use_gradient_checkpointing="unsloth",
        )

    return model, tokenizer
