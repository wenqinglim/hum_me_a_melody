# @title Setup MusicGen
from moods import get_moods
from astrology import get_planets
from music_generation import construct_musicgen_prompt, generate_music
from video_generation import get_video_gen_pipeline

import gradio as gr
# Import music generation model
import torch
import torchaudio

device = "cuda:0" if torch.cuda.is_available() else "cpu"


# @title Define wrapper functions for Gradio
def melody_to_composition(audio_file):
  """
  Wrapper function for Gradio

  Takes input audio and outputs generated music with an output message detailing its creative process.
  """
  sr, audio = audio_file
  moods, themes, functions = get_moods(audio)
  message = f"I'd describe your voice and melody as {', '.join(moods)}.\
    If that's not what you intended to sound like, then maybe practice singing more. "
  if themes:
    message += f"It has some {', '.join(themes)} vibes. "
  if functions:
    message += f"And it also sounds like it can be used as {', '.join(functions)} music. "

  planets = get_planets()
  if planets:
    message += f"\nThese are the planet(s) in my astrology sign today: {planets}.\
    I have composed a piece inspired by them. "

  musicgen_prompt = construct_musicgen_prompt(moods, themes, functions, planets, planets_instruments)

  wav = generate_music(musicgen_prompt, audio, sr)

  message += f"\nBased on your melody, I'm have created {musicgen_prompt}. Take a listen - what does it remind you of? "

  torch.cuda.empty_cache()


  return "composition.wav", moods, message


def text_to_muvi(user_input_txt, moods):
  """
  Wrapper function for Gradio

  Takes user input text and generates a video.
  Outputs a combined video with audio, and a message detailing its creative process.
  """
  video_prompt = f"{user_input_txt}, {', '.join(moods)} mood"
  # Generate video (+ 10.6GB of GPU RAM usage)
  pipe = get_video_gen_pipeline(device)
  output = pipe(
                prompt=video_prompt,
                negative_prompt="bad quality, worse quality, watermark",
                guidance_scale=1.0,
                num_frames=16,
                num_inference_steps=step
                )
  export_to_video(output.frames[0]*5, "video_only.mp4")
  torch.cuda.empty_cache()

  # Combine audio and video
  audio = ffmpeg.input("composition.wav")
  video = ffmpeg.input("video_only.mp4")


  music_video = ffmpeg.concat(video, audio, v=1, a=1)

  music_video.output("muvi.mp4").run(overwrite_output=True)

  message = f"Since the music sounds like '{user_input_txt}' to you,\
  I've created a video that captures that imagery in a {', '.join(moods)} mood.\
  Here's the masterpiece: A music video!"

  return "muvi.mp4", message



if __name__ == "__main__":
    print("Starting server...")
    demo = gr.Blocks()

    with demo:

        # 1. Get users to record a melody
        gr.Markdown(
        """
        # Hello World!
        Could you please hum me a melody?
        """)
        moods = gr.State([])
        audio_file = gr.Audio(type="numpy")
        b1 = gr.Button("Submit melody")
        message = gr.Textbox()

        # 2. Let users listen to generated music
        wav_path = gr.Audio(interactive=False)
        user_text = gr.Textbox(placeholder="What does the melody remind you of?")
        b2 = gr.Button("Submit interpretation")

        # 3. Show final music video
        message_final = gr.Textbox()
        muvi_path = gr.PlayableVideo()

        b1.click(melody_to_composition, inputs=audio_file, outputs=[wav_path, moods, message])
        b2.click(text_to_muvi, inputs=[user_text, moods], outputs=[muvi_path, message_final])

        demo.unload(lambda: print("unloading"))

    demo.launch(debug=True, height=1500)