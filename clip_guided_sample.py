from argparse import ArgumentParser
import os
from PIL import Image
import torch as th
import polars as pl
from glide_text2im.clip.model_creation import create_clip_model
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)
import re


def clean_text(text):
    text = text.strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r' ', '_', text)
    return text


def save_image(batch, key, caption):
    clean_caption = clean_text(caption)
    os.makedirs(f'./clip_figures/{clean_caption}', exist_ok=True)
    scaled = ((batch + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    image = Image.fromarray(reshaped.numpy())
    image.save(f'./clip_figures/{clean_caption}/{key}.png')
    with open(f'./clip_figures/{clean_caption}/{key}.txt', mode='w') as txtfile:
        txtfile.write(caption)


def main():
    has_cuda = th.cuda.is_available()
    device = th.device('cpu' if not has_cuda else 'cuda')

    # Create base model.
    options = model_and_diffusion_defaults()
    options['use_fp16'] = has_cuda
    options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
    model, diffusion = create_model_and_diffusion(**options)
    model.eval()
    if has_cuda:
        model.convert_to_fp16()
    model.to(device)
    model.load_state_dict(load_checkpoint('base', device))
    print('total base parameters', sum(x.numel() for x in model.parameters()))
    # Create CLIP model.
    clip_model = create_clip_model(device=device)
    clip_model.image_encoder.load_state_dict(load_checkpoint('clip/image-enc', device))
    clip_model.text_encoder.load_state_dict(load_checkpoint('clip/text-enc', device))

    # Create upsampler model.
    options_up = model_and_diffusion_defaults_upsampler()
    options_up['use_fp16'] = has_cuda
    options_up['timestep_respacing'] = 'fast27' # use 27 diffusion steps for very fast sampling
    model_up, diffusion_up = create_model_and_diffusion(**options_up)
    model_up.eval()
    if has_cuda:
        model_up.convert_to_fp16()
    model_up.to(device)
    model_up.load_state_dict(load_checkpoint('upsample', device))
    print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))

    batch_size = 1
    guidance_scale = 3.0

    # Tune this parameter to control the sharpness of 256x256 images.
    # A value of 1.0 is sharper, but sometimes results in grainy artifacts.
    upsample_temp = 0.997
    with open('./prompts.txt') as txtfile:
        data = [line.strip() for line in txtfile.readlines()]

    #data = pl.read_parquet('index.parquet').filter(pl.col('group') == group)

    for i, caption in enumerate(data):
        ##############################
        # Sample from the base model #
        ##############################

        # Create the text tokens to feed to the model.
        tokens = model.tokenizer.encode(caption)
        tokens, mask = model.tokenizer.padded_tokens_and_mask(
            tokens, options['text_ctx']
        )
        # Pack the tokens together into model kwargs.
        model_kwargs = dict(
            tokens=th.tensor([tokens] * batch_size, device=device),
            mask=th.tensor([mask] * batch_size, dtype=th.bool, device=device),
        )

        # Setup guidance function for CLIP model.
        cond_fn = clip_model.cond_fn([caption] * batch_size, guidance_scale)
       
        # Sample from the base model.
        model.del_cache()
        samples = diffusion.p_sample_loop(
            model,
            (batch_size, 3, options["image_size"], options["image_size"]),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
        )[:batch_size]
        model.del_cache()


        ##############################
        # Upsample the 64x64 samples #
        ##############################
        tokens = model_up.tokenizer.encode(caption)
        tokens, mask = model_up.tokenizer.padded_tokens_and_mask(
            tokens, options_up['text_ctx']
        )

        # Create the model conditioning dict.
        model_kwargs = dict(
            # Low-res image to upsample.
            low_res=((samples+1)*127.5).round()/127.5 - 1,

            # Text tokens
            tokens=th.tensor(
                [tokens] * batch_size, device=device
            ),
            mask=th.tensor(
                [mask] * batch_size,
                dtype=th.bool,
                device=device,
            ),
        )

        # Sample from the base model.
        model_up.del_cache()
        up_shape = (batch_size, 3, options_up["image_size"], options_up["image_size"])
        up_samples = diffusion_up.ddim_sample_loop(
            model_up,
            up_shape,
            noise=th.randn(up_shape, device=device) * upsample_temp,
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
        model_up.del_cache()
        save_image(up_samples, i, caption)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--group', type=int)
    main()
