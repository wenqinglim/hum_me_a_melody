import numpy as np
import ffmpeg

# Mood detection on audio
#@title Load the MTG Listening Models

# Thank you Dr Colton for the MTG model code snippets from lab notebooks.

from essentia.standard import AudioLoader, MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D
import essentia

embeddings_model = TensorflowPredictEffnetDiscogs(
    graphFilename="discogs-effnet-bs64-1.pb",
    output="PartitionedCall:1",
)

mood_classification_model = TensorflowPredict2D(
    graphFilename="mtg_jamendo_moodtheme-discogs-effnet-1.pb",
    output='model/Sigmoid',
)

# Functions for Using the Listening Models

all_tags = [
  "action", "adventure", "advertising",
  "background", "ballad",
  "calm", "children", "christmas", "commercial", "cool", "corporate",
  "dark", "deep", "documentary", "drama", "dramatic", "dream",
  "emotional", "energetic", "epic",
  "fast", "film", "fun", "funny",
  "game", "groovy",
  "happy", "heavy", "holiday", "hopeful",
  "inspiring",
  "love",
  "meditative", "melancholic", "melodic", "motivational", "movie",
  "nature",
  "party", "positive", "powerful",
  "relaxing", "retro", "romantic",
  "sad", "sexy", "slow", "soft", "soundscape", "space", "sport", "summer",
  "trailer", "travel",
  "upbeat", "uplifting"
]

# Tags categorised with help from ChatGPT (checked and refined by me)
# Moods
mood_tags = [
    "calm", "cool",
    "dark", "deep", "dramatic",
    "emotional", "energetic", "epic",
    "fast", "fun", "funny",
    "groovy",
    "happy", "heavy", "hopeful",
    "inspiring",
    "meditative", "melancholic", "motivational",
    "positive", "powerful",
    "relaxing", "romantic", "sad", "sexy", "slow", "soft",
     "upbeat", "uplifting"
]

# Themes
theme_tags = [
    "action", "adventure", "ballad",
    "children", "christmas",
     "dream", "film", "game",
    "holiday", "love", "movie",
    "nature", "party", "retro", "space","sport",
    "summer", "travel"
]

# Functions
function_tags = [
    "advertising", "background", "commercial",
    "corporate", "documentary", "drama",
    "soundscape", "trailer"
]

def get_mood_activations_dict(audio):
  embeddings = embeddings_model(audio)

  activations = mood_classification_model(embeddings)
  activation_avs = []
  for i in range(0, len(activations[0])):
    vals = [activations[j][i] for j in range(0, len(activations))]
    # Note - this does the averaging bit
    activation_avs.append(sum(vals)/len(vals))
  activations_dict = {}
  for ind, tag in enumerate(all_tags):
    activations_dict[tag] = activation_avs[ind]

  return activations_dict


def get_moods(audio):

  # Get mood activations
  mood_activations_dict = get_mood_activations_dict(essentia.array(audio))

  # Calculate upper threshold for outliers
  q1 = np.quantile(list(mood_activations_dict.values()), 0.25)
  q3 = np.quantile(list(mood_activations_dict.values()), 0.75)
  # Outliers defined as Q3 + 1.5*IQR
  outlier_threshold = q3 + (1.5*(q3-q1))

  # Select moods that are above outlier threshold (i.e. most prominent moods)
  tags = [tag for tag, score in mood_activations_dict.items() if (score >= outlier_threshold) & (tag != 'melodic')]
  moods = [tag for tag in tags if tag in mood_tags]
  themes = [tag for tag in tags if tag in theme_tags]
  functions = [tag for tag in tags if tag in function_tags]

  return moods, themes, functions