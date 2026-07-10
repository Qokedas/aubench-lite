"""Frozen host-side constants for CAM1K Beamforming Eval v1.

Everything here is grader/build infrastructure. The evaluated agent receives only
BRIEF.md, the raw visible capture, and the vendor export tensor.
"""

from __future__ import annotations

from typing import Final

SAMPLE_RATE_HZ: Final = 46_875
MICROPHONE_COUNT: Final = 1_024
SPEED_OF_SOUND_M_S: Final = 343.0
BLOCK_LENGTH: Final = 8_192
GRID_HEIGHT: Final = 24
GRID_WIDTH: Final = 32
BAND_COUNT: Final = 11
HORIZONTAL_FOV_DEG: Final = 70.42
VERTICAL_FOV_DEG: Final = 43.3
DYNAMIC_RANGE_DB: Final = 40.0

# The visible capture handed to the agent: the first 2.5 seconds of axcar1.
VISIBLE_SAMPLES: Final = 117_187

FREQUENCY_BANDS: Final = (
    (282.0, 315.0, 355.0),
    (355.0, 400.0, 447.0),
    (447.0, 500.0, 562.0),
    (562.0, 630.0, 708.0),
    (708.0, 800.0, 891.0),
    (891.0, 1_000.0, 1_122.0),
    (1_122.0, 1_250.0, 1_413.0),
    (1_413.0, 1_600.0, 1_778.0),
    (1_778.0, 2_000.0, 2_239.0),
    (2_239.0, 2_500.0, 2_818.0),
    (2_840.0, 4_000.0, 5_680.0),
)

BAND_CENTERS: Final = tuple(center for _, center, _ in FREQUENCY_BANDS)

# These two files are the only public raw CAM1K captures in set3 of the dataset Drive.
# The ROS bags in set1/set2/set4 do not contain microphone pressure samples.
SOURCE_CAPTURES: Final = {
    "axcar1": {
        "directory": "axcar1-10s-10m-c1-background",
        "sound_id": "11yAWA1VE8pdX7opFHhODCmIsZuBzkmrX",
        "metadata_id": "11zCvAevSe5C7MAs9G5YWTKJ3Og8nXKhb",
        "sound_bytes": 1_920_000_000,
        "sound_sha256": "40cf7cb7ff863bf63d1c15cdc2d0feb4f55fbaca2d68f332cdf8bbf54fc50154",
        "metadata_sha256": "3859305b70d46a78c64817fa9e7dc58127c6c536d103af468f8744d9f41459c1",
    },
    "axcar3": {
        "directory": "axcar3-10s-10m-c1-background-outside-engine",
        "sound_id": "12BfMpgW4uzWUs8BkJpcVhv7pN66BME0d",
        "metadata_id": "12CT0yifzqe-457iVsQSrJDsoFIBsLiDI",
        "sound_bytes": 1_920_000_000,
        "sound_sha256": "be8d0dc15141989706107a9f015523922a2767e6ec2606299bcff4df682942d2",
        "metadata_sha256": "f32acfe13a0545366ddb30df547b74024e7b8ce408388df4bc51f9f6d0e11424",
    },
}

DATASET_FOLDER_URL: Final = (
    "https://drive.google.com/drive/folders/1CJHbDfqtglHpCa12HLMzc4xfymDuhZBK"
)
