import librosa
import numpy as np


class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray):
        if self.always_apply:
            return self.apply(y)
        else:
            if np.random.rand() < self.p:
                return self.apply(y)
            else:
                return y

    def apply(self, y: np.ndarray):
        raise NotImplementedError


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        for trns in self.transforms:
            y = trns(y)
        return y


class OneOf:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        n_trns = len(self.transforms)
        trns_idx = np.random.choice(n_trns)
        trns = self.transforms[trns_idx]
        return trns(y)


class TimeStretch(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_rate=1.2):
        super().__init__(always_apply, p)

        self.max_rate = max_rate

    def apply(self, y: np.ndarray):
        rate = np.random.uniform(0, self.max_rate)
        augmented = librosa.effects.time_stretch(y, rate=rate)
        return augmented


class GaussianNoiseSNR(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5.0, max_snr=20.0):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y**2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise**2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented


class TimeShift(AudioTransform):
    def __init__(
        self,
        always_apply=False,
        p=0.5,
        max_shift_second=2,
        sr=48000,
        padding_mode="replace",
    ):
        super().__init__(always_apply, p)

        assert padding_mode in [
            "replace",
            "zero",
        ], "`padding_mode` must be either 'replace' or 'zero'"
        self.max_shift_second = max_shift_second
        self.sr = sr
        self.padding_mode = padding_mode

    def apply(self, y: np.ndarray, **params):
        shift = np.random.randint(
            -self.sr * self.max_shift_second, self.sr * self.max_shift_second
        )
        augmented = np.roll(y, shift)
        if self.padding_mode == "zero":
            if shift > 0:
                augmented[:shift] = 0
            else:
                augmented[shift:] = 0
        return augmented


class PitchShift(AudioTransform):
    """Shifting time axis"""

    def __init__(self, always_apply=False, p=0.5, max_rate=10, sr=16000):
        super(PitchShift, self).__init__(always_apply, p)
        self.max_rate = max_rate
        self.sr = sr

    def apply(self, data, return_sr=False):
        n_steps = np.random.randint(-self.max_rate, self.max_rate)
        augmented_sound = librosa.effects.pitch_shift(data, self.sr, n_steps)
        if return_sr:
            return augmented_sound, n_steps
        return augmented_sound


class VolumeControl(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, db_limit=10, mode="uniform"):
        super().__init__(always_apply, p)

        assert mode in [
            "uniform",
            "fade",
            "fade",
            "cosine",
            "sine",
        ], "`mode` must be one of 'uniform', 'fade', 'cosine', 'sine'"

        self.db_limit = db_limit
        self.mode = mode

    def apply(self, y: np.ndarray):
        db = np.random.uniform(-self.db_limit, self.db_limit)
        if self.mode == "uniform":
            db_translated = 10 ** (db / 20)
        elif self.mode == "fade":
            lin = np.arange(len(y))[::-1] / (len(y) - 1)
            db_translated = 10 ** (db * lin / 20)
        elif self.mode == "cosine":
            cosine = np.cos(np.arange(len(y)) / len(y) * np.pi * 2)
            db_translated = 10 ** (db * cosine / 20)
        else:
            sine = np.sin(np.arange(len(y)) / len(y) * np.pi * 2)
            db_translated = 10 ** (db * sine / 20)
        augmented = y * db_translated
        return augmented


class ShiftingTime(AudioTransform):
    """Shifting time axis"""

    def __init__(self, always_apply=False, p=0.5):
        super(ShiftingTime, self).__init__(always_apply, p)

    def apply(self, data):
        shift = np.random.randint(0, len(data) // 2)
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift

        augmented_sound = np.roll(data, shift)
        # Set to silence for heading/ tailing
        if shift > 0:
            augmented_sound[:shift] = 0
        else:
            augmented_sound[shift:] = 0
        return augmented_sound
