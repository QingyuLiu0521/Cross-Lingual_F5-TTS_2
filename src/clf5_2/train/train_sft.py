# training script.

import os
from importlib.resources import files

import hydra
from omegaconf import OmegaConf

from clf5_2.model import CFM_SFT, Trainer_SFT
from clf5_2.model.dataset import load_dataset
from clf5_2.model.utils import get_tokenizer


os.chdir(str(files("clf5_2").joinpath("../..")))  # change working directory to root of project (local editable)


@hydra.main(version_base="1.3", config_path=str(files("clf5_2").joinpath("configs")), config_name=None)
def main(model_cfg):
    model_cls = hydra.utils.get_class(f"clf5_2.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    tokenizer = model_cfg.model.tokenizer
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
    prefix_text_pad = bool(model_cfg.model.get("prefix_text_pad", False))
    prompt_token = bool(model_cfg.model.get("prompt_token", False))
    eos_marker = model_cfg.model.get("eos_marker", None)
    use_total_text = bool(model_cfg.model.get("use_total_text", False))

    if use_total_text and prefix_text_pad:
        raise ValueError("use_total_text and prefix_text_pad cannot both be True")

    exp_name = f"{model_cfg.model.name}_{mel_spec_type}_{model_cfg.model.tokenizer}_{model_cfg.datasets.name}_gp_finetune_f5Emilia_fixed"
    wandb_resume_id = None

    # set text tokenizer
    if tokenizer != "custom":
        tokenizer_path = model_cfg.datasets.name
    else:
        tokenizer_path = model_cfg.model.tokenizer_path
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer, sft=True)

    # set model
    model = CFM_SFT(
        transformer=model_cls(
            **model_arc,
            text_num_embeds=vocab_size + (1 if prompt_token else 0),
            mel_dim=model_cfg.model.mel_spec.n_mel_channels,
        ),
        mel_spec_kwargs=model_cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
        prefix_text_pad=prefix_text_pad,
        prompt_token=prompt_token,
        eos_marker=eos_marker,
    )

    # init trainer
    trainer = Trainer_SFT(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=str(files("clf5_2").joinpath(f"../../{model_cfg.ckpts.save_dir}")),
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_project="CFM-TTS",
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
        tokenizer=tokenizer,
        use_total_text=use_total_text,
    )

    train_dataset = load_dataset(model_cfg.datasets.name, tokenizer, sft=True, mel_spec_kwargs=model_cfg.model.mel_spec, target_hour=getattr(model_cfg.datasets, "target_hour", None))
    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
