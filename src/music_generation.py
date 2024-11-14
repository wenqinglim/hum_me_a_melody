# Import music generation model
import torch
import torchaudio


from audiocraft.models import MusicGen
from audiocraft.utils.notebook import display_audio
from audiocraft.data.audio import audio_write

def construct_musicgen_prompt(moods, themes, functions, planets, planets_instruments):
  moods_str = ", ".join(moods)

  prompt = f"a {moods_str} orchestral piece"
  if themes:
    prompt += f" with {', '.join(themes)} themes"
  for planet in planets:
    planet_instr = planets_instruments[planet]['instruments']
    planet_sound = planets_instruments[planet]['sound']
    planet_str = f"{', '.join(planet_instr)} with {str.lower(planet_sound)}"
    prompt += f". {planet_str}"
  if functions:
    prompt += f"perfect for {', '.join(functions)}"
  return prompt


def generate_music(musicgen_prompt, audio, sr):
  """
  Generates music using pre-trained MusicGen model
  """
  model = MusicGen.get_pretrained('melody')
  model.set_generation_params(duration=8)  # generate 8 seconds.

  # generates using the melody from the given audio and the provided descriptions.
  wav_gpu = model.generate_with_chroma([musicgen_prompt], torch.tensor(audio).float().reshape(1, 1, -1), sr)[0]
  wav = wav_gpu.cpu()
  del wav_gpu

  # Will save under composition.wav, with loudness normalization at -14 db LUFS.
  audio_write(f'composition', wav, model.sample_rate, strategy="loudness")

  return wav