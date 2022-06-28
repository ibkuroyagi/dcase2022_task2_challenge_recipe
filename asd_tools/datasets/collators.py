import numpy as np
import torch


class WaveCollator(object):
    """Wave form data's collator."""

    def __init__(
        self,
        sf=16000,
        sec=3,
        pos_machine="fan",
        shuffle=True,
        use_target=False,
        use_is_normal=False,
    ):
        """Initialize customized collator for PyTorch DataLoader."""
        self.sf = sf
        self.sec = sec
        self.max_frame = int(sf * sec)
        self.pos_machine = pos_machine
        self.shuffle = shuffle
        self.use_target = use_target
        self.use_is_normal = use_is_normal
        self.rng = np.random.default_rng()

    def __call__(self, batch):
        """Convert into batch tensors."""
        wave_batch, machine_batch, section_batch = [], [], []
        if self.use_is_normal:
            is_normal_batch = []
        if self.use_target:
            is_pos_ta_batch = []
        for b in batch:
            start_frame = (
                self.rng.integers(max(len(b["wave"]) - self.max_frame, 1), size=1)[0]
                if self.shuffle
                else 0
            )
            wave_batch.append(
                torch.tensor(
                    b["wave"][start_frame : start_frame + self.max_frame],
                    dtype=torch.float,
                )
            )
            machine_batch.append(int(b["machine"] == self.pos_machine))
            section_batch.append(b["section"])
            if self.use_is_normal:
                is_normal_batch.append(b["is_normal"])
            if self.use_target:
                is_pos_ta_batch.append(
                    int(
                        (b["machine"] == self.pos_machine) and (b["domain"] == "target")
                    )
                )
        items = {
            "wave": torch.stack(wave_batch),
            "machine": torch.tensor(machine_batch, dtype=torch.float),
            "section": torch.tensor(
                np.array(section_batch).flatten(), dtype=torch.long
            ),
        }
        if self.use_is_normal:
            items["is_normal"] = np.array(is_normal_batch)
        if self.use_target:
            items["is_pos_ta"] = torch.tensor(is_pos_ta_batch, dtype=torch.float)
        return items


class WaveEvalCollator(object):
    """Customized collator for Pytorch DataLoader for feat form data in evaluation."""

    def __init__(
        self,
        sf=16000,
        sec=3,
        n_split=3,
        use_domain=False,
        is_label=False,
        is_dcase2022=False,
    ):
        """Initialize customized collator for PyTorch DataLoader."""
        self.sf = sf
        self.sec = sec
        self.max_frames = int(sf * sec)
        self.n_split = n_split
        self.use_domain = use_domain
        self.is_label = is_label
        self.is_dcase2022 = is_dcase2022

    def __call__(self, batch):
        """Convert into batch tensors."""
        waves = [b["wave"] for b in batch]
        items = {}
        if self.n_split == 1:
            wave_batch = [torch.tensor(wave, dtype=torch.float) for wave in waves]
            items["X0"] = torch.stack(wave_batch)
        else:
            frame_lengths = np.array([wave.shape[0] for wave in waves])
            hop_size = np.array(
                [
                    max((frame_length - self.max_frames) // (self.n_split - 1), 1)
                    for frame_length in frame_lengths
                ]
            )
            start_frames = np.array(
                [(hop_size * i).astype(np.int64) for i in range(self.n_split - 1)]
                + [frame_lengths - self.max_frames]
            )
            end_frames = start_frames + self.max_frames

            for i, (start_frame, end_frame) in enumerate(zip(start_frames, end_frames)):
                wave_batch = [
                    torch.tensor(wave[start_frame[j] : end_frame[j]], dtype=torch.float)
                    for j, wave in enumerate(waves)
                ]
                items[f"X{i}"] = torch.stack(wave_batch)
        if self.is_dcase2022:
            items["machine"] = np.array([b["machine"] for b in batch])
            items["section"] = np.array([b["section"] for b in batch])
            items["path"] = np.array([b["path"] for b in batch])
            if self.is_label:
                items["is_normal"] = np.array([b["is_normal"] for b in batch])
            return items

        if self.use_domain:
            domains = [
                "source" if str(b["domain"]) == "source" else "target" for b in batch
            ]
            items["domain"] = np.array(domains)
        if self.is_label:
            states = [b["state"] for b in batch]
            label = []
            state_batch = []
            for state in states:
                if str(state, "utf-8") == "normal":
                    label.append("normal")
                    state_batch.append(1)
                elif str(state, "utf-8") == "anomaly":
                    label.append("anomaly")
                    state_batch.append(0)
            items["label"] = np.array(label)
            items["is_normal"] = torch.tensor(state_batch, dtype=torch.float).unsqueeze(
                1
            )
        return items
