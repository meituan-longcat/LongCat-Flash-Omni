import os

########
# -v
########
CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
IMAGE_PAD_TOKEN_INDEX = -201
DEFAULT_IMAGE_TOKEN = "<image>"
IMAGE_PAD_TOKEN = "<|image_pad|>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"
MAX_IMAGE_FRAME_NUM = int(os.getenv("MAX_IMAGE_FRAME_NUM", 24))
MAX_CONCAT_SAMPLE_NUM = int(os.getenv("MAX_CONCAT_SAMPLE_NUM", 20))
DEFAULT_GEN_IM_START_TOKEN = "<img>"
DEFAULT_GEN_IM_END_TOKEN = "</img>"

DEFAULT_SLICE_START_TOKEN = "[PLACEHOLDER_0]"
DEFAULT_SLICE_END_TOKEN = "[PLACEHOLDER_1]"

DEFAULT_AR_START_TOKEN = "<im_start>"
DEFAULT_AR_END_TOKEN = "<im_end>"

EOS_ID_FOR_PREPROCESS = int(os.getenv("EOS_ID_FOR_PREPROCESS", 2))

TOKEN_TYPE_NON_PROMPT = 31
TOKEN_TYPE_PROMPT = 32


########
# -s
########
# audio tokens
AUDIO_BOS_TOKEN = "<|audio|>"
AUDIO_EOS_TOKEN = "<|/audio|>"
AUDIO_PAD_TOKEN = "<|audio_pad|>"
CODEC_PAD_TOKEN = "<|codec_pad|>"
TMP_PAUSE_TOKEN = "</pause>"

SYSTEM_BOS_TOKEN = "<begin-of-system>"
USER_BOS_TOKEN = "<begin-of-user>"
ASSISTANT_BOS_TOKEN = "<begin-of-assistant>"
VOICE_ASSISTANT_BOS_TOKEN = "<begin-of-voice-assistant>"

VOICE_ASSISTANT_PREFIX = "VOICE ASSISTANT:"
ASSISTANT_PREFIX = "ASSISTANT:"

AUDIO_INPUT_FRAME_SEC = 0.08  # The duration of each input frame is 80 milliseconds.

TEXT2TEXT_SFT_DATA_TYPE_ID = 10  # Text input, text output
TEXT2CODEC_SFT_DATA_TYPE_ID = 11  # Text input, discrete speech output
AUDIO2TEXT_SFT_DATA_TYPE_ID = 12  # Continuous speech input, text output
AUDIO2CODEC_SFT_DATA_TYPE_ID = 13  # Continuous speech input, discrete speech output
CODEC2TEXT_SFT_DATA_TYPE_ID = 14  # Discrete speech input, text output
CODEC2CODEC_SFT_DATA_TYPE_ID = 15  # Discrete speech input, discrete speech output

SEQ_TYPE_TEXT = 1
SEQ_TYPE_AUDIO = 2
SEQ_TYPE_ASR = 3

PAD_FLAG_ID = 0
SYSTEM_FLAG_ID = 1
USER_FLAG_ID = 2
ASSISTANT_FLAG_ID = 3

CODEC_EOS_ID = 2
CODEC_PAD_ID = 3
NUM_CODEC_PLACEHOLDERS = 32

ROLE_FLAG_IDS = {
    "pad": PAD_FLAG_ID,
    "system": SYSTEM_FLAG_ID,
    "function": SYSTEM_FLAG_ID,
    "user": USER_FLAG_ID,
    "tool": USER_FLAG_ID,
    "assistant": ASSISTANT_FLAG_ID,
    "voice assistant": ASSISTANT_FLAG_ID,
}

# mask start index
MASK_START_IDX = 110
# eos index
EOD_INDEX = 2


########
# -omni
########
IMAGE_TEXT2TEXT_SFT_DATA_TYPE_ID = 51  # Image and text input, text output
IMAGE_CODEC2CODEC_SFT_DATA_TYPE_ID = (
    52  # Image and discrete speech input, speech output
)
IMAGE_AUDIO2CODEC_SFT_DATA_TYPE_ID = (
    53  # Image and continuous speech input, speech output
)
IMAGE_CODEC2TEXT_SFT_DATA_TYPE_ID = 54  # Image and discrete speech input, text output
IMAGE_AUDIO2TEXT_SFT_DATA_TYPE_ID = 55  # Image and continuous speech input, text output

VIDEO_TEXT2TEXT_SFT_DATA_TYPE_ID = 61  # Video and text input, text output
VIDEO_CODEC2CODEC_SFT_DATA_TYPE_ID = (
    62  # Video and discrete speech input, discrete speech output
)
VIDEO_AUDIO2CODEC_SFT_DATA_TYPE_ID = (
    63  # Video and continuous speech input, discrete speech output
)
VIDEO_CODEC2TEXT_SFT_DATA_TYPE_ID = 64  # Video and discrete speech input, text output
VIDEO_AUDIO2TEXT_SFT_DATA_TYPE_ID = 65  # Video and continuous speech input, text output
MULTIMODAL_INTERLEAVED_SFT_DATA_TYPE_ID = 70  # Interleaved audio and video
