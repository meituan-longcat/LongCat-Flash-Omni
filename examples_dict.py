def get_test_case_dict():
    text_case = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Is there anything delicious or fun to do in Beijing?",
                }
            ],
        }
    ]

    audio_case = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": "examples/audio_case.wav"},
                {"type": "text", "text": "What can you hear in the audio?"},
            ],
        }
    ]

    image_case = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "examples/image_case.png", "meta": {}},
                {"type": "text", "text": "What can you see in the image?"},
            ],
        }
    ]

    text_video_case = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "examples/wurenche_3s.mp4",
                    "meta": {
                        "read_image_fps": 1,
                        "bound": [2.0, 5.0],
                        "frame_sample_strategy": None,
                        "average_sample_num_frame_per_video": 8,
                        "sequential_sample_num_frame_per_sec": 1,
                        "sequential_max_num_sample_frame": 32,
                    },
                },
                {"type": "text", "text": "视频里出现了哪些物体？"},
            ],
        }
    ]

    video_audio_case = [
        {
            "role": "user",
            "content": [
                {
                    "type": "multimodal_interleaved",
                    "multimodal_interleaved": {
                        "video": "examples/wurenche_3s.mp4",
                        "audio": "examples/audio_question_3s.wav"
                        },
                    "meta": {
                        "frame_sample_strategy": 'sequential',
                        "average_sample_num_frame_per_video": 8,
                        "sequential_sample_num_frame_per_sec": 1,
                        "use_audio_track": False,
                        "time_offset": 0.0,
                        "video_frame_pre_resize": False,
                        "video_max_resolution": 720,
                        "sequential_max_num_sample_frame": -1,
                        "bound": None
                    }
                }
            ]
        }
    ]


    video_case = [
        {
            "role": "user",
            "content": [
                {
                    "type": "multimodal_interleaved",
                    "multimodal_interleaved": {
                        "video": "examples/wurenche_3s_interleaved.mp4",
                    },
                    "meta": {
                        "frame_sample_strategy": 'sequential',
                        "average_sample_num_frame_per_video": 8,
                        "sequential_sample_num_frame_per_sec": 1,
                        "use_audio_track": True,
                        "time_offset": 0.0,
                        "video_frame_pre_resize": False,
                        "video_max_resolution": 720,
                        "sequential_max_num_sample_frame": -1,
                        "bound": None
                    }
                }
            ]
        }
    ]

    result_dict = {
        "text_case": text_case,
        "audio_case": audio_case,
        "image_case": image_case,
        "text_video_case": text_video_case,
        "video_audio_case": video_audio_case,
        "video_case": video_case,
    }
    return result_dict
