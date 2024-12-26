import sys
import io, os, stat
import subprocess
import random
import uuid
import time
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir

import gradio as gr
from scipy.io.wavfile import write
import re

class ModelManager:
    def __init__(self):
        self.current_model = None
        self.current_model_type = "basic"
        self.load_model("basic")
    
    def load_model(self, model_type):
        """Load and return the specified model type"""
        print(f"Loading {model_type} model...")
        config = XttsConfig()
        
        if model_type == "basic":
            config.load_json("models/kss_ft/config.json")
            checkpoint_path = "models/base/model.pth"
        elif model_type == "single_speaker":
            config.load_json("models/kss_ft/config.json")
            checkpoint_path = "models/kss_ft/model.pth"
        elif model_type == "multi_speaker":
            config.load_json("models/ksponspeech_ft/config.json")
            checkpoint_path = "models/ksponspeech_ft/model.pth"
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_path=checkpoint_path,
            vocab_path="models/base/vocab.json",
            eval=True,
            use_deepspeed=False
        )
        model.cuda()
        
        self.current_model = model
        self.current_model_type = model_type
        return f'Model set to: {model_type.replace("_", " ").title()}'

    def get_model(self):
        """Return the currently loaded model"""
        return self.current_model

def predict(
    prompt,
    audio_file_pth,
    voice_cleanup,
    model_type,
    model_manager
):
    speaker_wav = audio_file_pth
    model = model_manager.get_model()

    # Voice cleanup for microphone input
    if voice_cleanup:
        try:
            out_filename = speaker_wav + str(uuid.uuid4()) + ".wav"
            shell_command = f"ffmpeg -y -i {speaker_wav} -af lowpass=8000,highpass=75,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02,areverse,silenceremove=start_periods=1:start_silence=0:start_threshold=0.02 {out_filename}".split()
            subprocess.run(shell_command, capture_output=False, text=True, check=True)
            speaker_wav = out_filename
            print("Filtered microphone input")
        except subprocess.CalledProcessError:
            print("Error: failed filtering, using original microphone input")

    if len(prompt) < 2:
        gr.Warning("Please give a longer prompt text")
        return None, None, None, None
    
    if len(prompt) > 200:
        gr.Warning("Text length limited to 200 characters. Please try shorter text.")
        return None, None, None, None

    try:
        metrics_text = ""
        t_latent = time.time()

        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=speaker_wav,
            gpt_cond_len=30,
            gpt_cond_chunk_len=4,
            max_ref_length=60
        )

        print("Generating new audio...")
        t0 = time.time()
        out = model.inference(
            prompt,
            "ko",  # Fixed to Korean
            gpt_cond_latent,
            speaker_embedding,
            repetition_penalty=5.0,
            temperature=0.75,
        )
        
        inference_time = time.time() - t0
        print(f"Time to generate audio: {round(inference_time*1000)} milliseconds")
        metrics_text += f"Time to generate audio: {round(inference_time*1000)} milliseconds\n"
        
        real_time_factor = (time.time() - t0) / out['wav'].shape[-1] * 24000
        print(f"Real-time factor (RTF): {real_time_factor}")
        metrics_text += f"Real-time factor (RTF): {real_time_factor:.2f}\n"
        os.makedirs("results", exist_ok=True)
        torchaudio.save("results/output.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)

        # make a small plot along with a spectrogram of a result
        audio = out["wav"]
        audio = audio / np.max(np.abs(audio))
        f, t, Sxx = spectrogram(audio, 24000, nperseg=1024, noverlap=512)
        plt.figure(figsize=(10, 4))
        plt.subplot(2, 1, 1)
        plt.plot(audio)
        plt.title("Waveform")
        plt.subplot(2, 1, 2)
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-6), shading='gouraud')
        plt.title("Spectrogram")
        plt.tight_layout()
        plt.savefig("results/plot.png")
        plt.close()

    except Exception as e:
        print("Error during inference:", str(e))
        gr.Warning("An error occurred during generation. Please try again.")
        return None, None, None, None
    
    return (
        "results/plot.png",
        "results/output.wav",
        metrics_text,
        speaker_wav,
    )

# Initialize the model manager
model_manager = ModelManager()

# Gradio Interface
title = "Korean TTS"

description = """
<br/>
This demo runs a Korean-optimized version of TTS, supporting high-quality Korean text-to-speech with voice cloning capabilities.
Simply input Korean text and choose available speaker or provide a reference voice to clone.
<br/>
"""

# Interface definition
with gr.Blocks(analytics_enabled=False) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)
    
    with gr.Row():
        with gr.Column():
            load_model_gr = gr.Dropdown(
                label="Model Type",
                info="Select the model type to use",
                choices=[
                    "basic",
                    "single_speaker",
                    "multi_speaker"
                ],
                value="basic"
            )
            model_info = gr.Markdown('Model set to: Basic')
            
            input_text_gr = gr.Textbox(
                label="Korean Text",
                info="Enter Korean text (up to 200 characters)",
                value="안녕하세요, 반갑습니다."
            )
            ref_gr = gr.Audio(
                label="Reference Audio",
                type="filepath",
                value="samples/kss.wav"
            )
            clean_ref_gr = gr.Checkbox(
                label="Clean Reference Audio",
                value=False,
                info="Apply noise reduction to reference audio"
            )
            tts_button = gr.Button("Generate", elem_id="generate-btn")

        with gr.Column():
            image_gr = gr.Image(label="Waveform Visualization")
            audio_gr = gr.Audio(label="Generated Audio", autoplay=True)
            out_text_gr = gr.Text(label="Performance Metrics")

    def update_model(model_type):
        return model_manager.load_model(model_type)

    load_model_gr.change(
        update_model,
        inputs=[load_model_gr],
        outputs=[model_info]
    )

    tts_button.click(
        predict,
        inputs=[
            input_text_gr,
            ref_gr,
            clean_ref_gr,
            load_model_gr,
            gr.State(model_manager)
        ],
        outputs=[
            image_gr,
            audio_gr,
            out_text_gr,
            ref_gr
        ]
    )

demo.queue()
demo.launch(debug=True, show_api=True)