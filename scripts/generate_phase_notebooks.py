import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def md_cell(text):
    text = textwrap.dedent(text).strip()
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line if line.endswith("\n") else line + "\n" for line in text.splitlines()],
    }


def code_cell(code):
    code = textwrap.dedent(code).strip()
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line if line.endswith("\n") else line + "\n" for line in code.splitlines()],
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


common_header = """# Learnable MSFA Research Track

This notebook is part of the 10-day publication-oriented extension track.

## Run discipline
- Keep one experiment purpose per notebook.
- Save metrics and checkpoints to the configured project root.
- Do not silently change data splits or loss definitions across notebooks.
- When a result becomes final, export both numeric artifacts and a figure/table asset.
"""


notebooks = {}


def colab_mount_cell():
    return code_cell(
        """
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            print("Google Drive mounted.")
        except Exception as exc:
            print("Drive mount skipped:", exc)
        """
    )

notebooks["01_phase1_data_protocol.ipynb"] = notebook(
    [
        md_cell(
            """
            # Phase 1: Data Protocol And Patch Dataset

            Goal: build a reproducible 16-band CAVE patch dataset with a fixed scene split.

            Deliverables:
            - `dataset_patches.npz`
            - `split_metadata.json`
            - one qualitative sanity-check figure
            """
        ),
        md_cell(common_header),
        colab_mount_cell(),
        code_cell(
            """
            import json
            import random
            import zipfile
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            from PIL import Image

            SEED = 42
            random.seed(SEED)
            np.random.seed(SEED)

            PROJECT_ROOT = Path("/content/drive/MyDrive/Msa-Osp")
            ZIP_PATH = PROJECT_ROOT / "complete_ms_data.zip"
            UNZIP_DIR = Path("/content/unzipped_data")
            PATCH_PATH = PROJECT_ROOT / "dataset_patches.npz"
            SPLIT_PATH = PROJECT_ROOT / "split_metadata.json"

            PATCH_SIZE = 128
            PATCHES_PER_SCENE = 40
            BAND_COUNT = 16
            TRAIN_SCENES = 24

            PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
            print("Project root:", PROJECT_ROOT)
            """
        ),
        code_cell(
            """
            if not UNZIP_DIR.exists():
                with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
                    zip_ref.extractall(UNZIP_DIR)

            scene_names = sorted(d.name for d in UNZIP_DIR.iterdir() if d.is_dir())
            print("Total scenes:", len(scene_names))
            print("First five scenes:", scene_names[:5])
            """
        ),
        code_cell(
            """
            def load_scene(scene_root, band_count=BAND_COUNT):
                subdirs = [d for d in scene_root.iterdir() if d.is_dir()]
                if not subdirs:
                    raise ValueError(f"No nested folder found inside {scene_root}")

                scene_dir = subdirs[0]
                band_files = sorted(
                    [p for p in scene_dir.iterdir() if p.suffix.lower() == ".png" and "_ms_" in p.name],
                    key=lambda p: int(p.stem.split("_ms_")[1]),
                )[:band_count]

                bands = []
                for path in band_files:
                    image = np.array(Image.open(path), dtype=np.float32)
                    if image.ndim == 3:
                        image = image[:, :, 0]
                    bands.append(image)

                cube = np.stack(bands, axis=-1)
                cube /= 65535.0 if cube.max() > 255 else 255.0
                return cube.astype(np.float32)


            def extract_random_patches(cube, patch_size=PATCH_SIZE, patches_per_scene=PATCHES_PER_SCENE):
                h, w, _ = cube.shape
                patches = []
                for _ in range(patches_per_scene):
                    top = random.randint(0, h - patch_size)
                    left = random.randint(0, w - patch_size)
                    patches.append(cube[top:top + patch_size, left:left + patch_size, :])
                return patches


            def to_rgb(cube):
                rgb = cube[:, :, [5, 10, 15]]
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
                return rgb
            """
        ),
        code_cell(
            """
            train_scenes = scene_names[:TRAIN_SCENES]
            val_scenes = scene_names[TRAIN_SCENES:]

            train_patches = []
            val_patches = []

            for scene_name in train_scenes:
                cube = load_scene(UNZIP_DIR / scene_name)
                train_patches.extend(extract_random_patches(cube))

            for scene_name in val_scenes:
                cube = load_scene(UNZIP_DIR / scene_name)
                val_patches.extend(extract_random_patches(cube))

            train_data = np.asarray(train_patches, dtype=np.float32)
            val_data = np.asarray(val_patches, dtype=np.float32)

            PATCH_PATH.parent.mkdir(parents=True, exist_ok=True)
            SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(PATCH_PATH, train=train_data, val=val_data)
            SPLIT_PATH.write_text(
                json.dumps(
                    {
                        "seed": SEED,
                        "band_count": BAND_COUNT,
                        "patch_size": PATCH_SIZE,
                        "patches_per_scene": PATCHES_PER_SCENE,
                        "train_scenes": train_scenes,
                        "val_scenes": val_scenes,
                    },
                    indent=2,
                )
            )

            print("Train shape:", train_data.shape)
            print("Val shape:", val_data.shape)
            """
        ),
        code_cell(
            """
            example = train_data[0]
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(to_rgb(example))
            plt.title("Example fake RGB")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.plot(example[example.shape[0] // 2, example.shape[1] // 2, :])
            plt.title("Center-pixel spectrum")
            plt.xlabel("Band")
            plt.ylabel("Normalized intensity")
            plt.tight_layout()
            plt.show()
            """
        ),
    ]
)

notebooks["02_phase2_fixed_msfa_baseline.ipynb"] = notebook(
    [
        md_cell(
            """
            # Phase 2: Fixed-MSFA Baseline

            Goal: train a strong fixed-mask baseline and log paper-ready metrics.

            Deliverables:
            - `msfa_dataset_4x4.npz`
            - `baseline_history.csv`
            - `unet_baseline_best.pth`
            - baseline qualitative figure
            """
        ),
        md_cell(common_header),
        colab_mount_cell(),
        code_cell(
            """
            import csv
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, Dataset

            PROJECT_ROOT = Path("/content/drive/MyDrive/Msa-Osp")
            PATCH_PATH = PROJECT_ROOT / "dataset_patches.npz"
            MSFA_DATA_PATH = PROJECT_ROOT / "msfa_dataset_4x4.npz"
            HISTORY_PATH = PROJECT_ROOT / "baseline_history.csv"
            CKPT_PATH = PROJECT_ROOT / "unet_baseline_best.pth"
            FIG_PATH = PROJECT_ROOT / "baseline_example.png"

            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            BATCH_SIZE = 2
            EPOCHS = 40
            LR = 1e-4
            BASE = 32
            BAND_COUNT = 16
            """
        ),
        code_cell(
            """
            data = np.load(PATCH_PATH)
            train_target = data["train"]
            val_target = data["val"]

            def create_fixed_mask(h, w, bands=BAND_COUNT):
                tile = np.arange(bands).reshape(4, 4)
                mask = np.zeros((h, w, bands), dtype=np.float32)
                for i in range(h):
                    for j in range(w):
                        mask[i, j, tile[i % 4, j % 4]] = 1.0
                return mask

            mask = create_fixed_mask(train_target.shape[1], train_target.shape[2])

            def apply_msfa(cubes, mask_3d):
                mosaiced = np.sum(cubes * mask_3d[None, ...], axis=-1)
                inputs = mosaiced[..., None] * mask_3d[None, ...]
                return inputs.astype(np.float32), cubes.astype(np.float32)

            train_input, train_target = apply_msfa(train_target, mask)
            val_input, val_target = apply_msfa(val_target, mask)

            MSFA_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                MSFA_DATA_PATH,
                train_input=train_input,
                train_target=train_target,
                val_input=val_input,
                val_target=val_target,
            )
            """
        ),
        code_cell(
            """
            class MSFADataset(Dataset):
                def __init__(self, inputs, targets):
                    self.inputs = inputs
                    self.targets = targets

                def __len__(self):
                    return len(self.inputs)

                def __getitem__(self, idx):
                    x = torch.from_numpy(self.inputs[idx]).permute(2, 0, 1).float()
                    y = torch.from_numpy(self.targets[idx]).permute(2, 0, 1).float()
                    return x, y

            class DoubleConv(nn.Module):
                def __init__(self, in_ch, out_ch):
                    super().__init__()
                    self.block = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_ch, out_ch, 3, padding=1),
                        nn.ReLU(inplace=True),
                    )

                def forward(self, x):
                    return self.block(x)

            class UNet2D(nn.Module):
                def __init__(self, in_ch=16, out_ch=16, base=32):
                    super().__init__()
                    self.enc1 = DoubleConv(in_ch, base)
                    self.pool1 = nn.MaxPool2d(2)
                    self.enc2 = DoubleConv(base, base * 2)
                    self.pool2 = nn.MaxPool2d(2)
                    self.bottleneck = DoubleConv(base * 2, base * 4)
                    self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
                    self.dec2 = DoubleConv(base * 4, base * 2)
                    self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
                    self.dec1 = DoubleConv(base * 2, base)
                    self.final = nn.Conv2d(base, out_ch, 1)

                def forward(self, x):
                    e1 = self.enc1(x)
                    e2 = self.enc2(self.pool1(e1))
                    b = self.bottleneck(self.pool2(e2))
                    d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
                    d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
                    return self.final(d1)

            def compute_psnr(pred, target, eps=1e-8):
                mse = torch.mean((pred - target) ** 2)
                return 20 * torch.log10(1.0 / torch.sqrt(mse + eps))

            def spectral_angle_mapper(pred, target, eps=1e-8):
                dot = torch.sum(pred * target, dim=1)
                pred_norm = torch.norm(pred, dim=1)
                target_norm = torch.norm(target, dim=1)
                cos_theta = torch.clamp(dot / (pred_norm * target_norm + eps), -1 + eps, 1 - eps)
                return torch.rad2deg(torch.acos(cos_theta)).mean()

            def spectral_angle_loss(pred, target, eps=1e-8):
                dot = torch.sum(pred * target, dim=1)
                pred_norm = torch.norm(pred, dim=1)
                target_norm = torch.norm(target, dim=1)
                cos_theta = torch.clamp(dot / (pred_norm * target_norm + eps), -1 + eps, 1 - eps)
                return torch.acos(cos_theta).mean()

            def compute_rgb_ssim(pred, target):
                if not HAS_SSIM:
                    return float("nan")
                pred_np = pred.detach().cpu().numpy()
                target_np = target.detach().cpu().numpy()
                scores = []
                for i in range(pred_np.shape[0]):
                    p = np.transpose(pred_np[i, [5, 10, 15]], (1, 2, 0))
                    t = np.transpose(target_np[i, [5, 10, 15]], (1, 2, 0))
                    data_range = max(float(t.max() - t.min()), 1e-8)
                    scores.append(ssim_fn(t, p, data_range=data_range, channel_axis=2))
                return float(np.mean(scores))
            """
        ),
        code_cell(
            """
            train_loader = DataLoader(MSFADataset(train_input, train_target), batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(MSFADataset(val_input, val_target), batch_size=BATCH_SIZE, shuffle=False)

            model = UNet2D(base=BASE).to(DEVICE)
            criterion = nn.L1Loss()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)

            best_psnr = -float("inf")
            history = []

            for epoch in range(1, EPOCHS + 1):
                model.train()
                train_loss = 0.0
                for x, y in train_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    pred = model(x)
                    loss = criterion(pred, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                model.eval()
                val_psnr = 0.0
                val_sam = 0.0
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(DEVICE), y.to(DEVICE)
                        pred = model(x)
                        val_psnr += compute_psnr(pred, y).item()
                        val_sam += spectral_angle_mapper(pred, y).item()

                train_loss /= len(train_loader)
                val_psnr /= len(val_loader)
                val_sam /= len(val_loader)
                history.append({"epoch": epoch, "train_l1": train_loss, "val_psnr": val_psnr, "val_sam_deg": val_sam})
                print(f"Epoch {epoch:02d} | train L1 {train_loss:.4f} | val PSNR {val_psnr:.2f} dB | val SAM {val_sam:.2f} deg")

                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_val_psnr": best_psnr,
                        },
                        CKPT_PATH,
                    )

            HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(HISTORY_PATH, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
                writer.writeheader()
                writer.writerows(history)
            """
        ),
    ]
)

notebooks["03_phase3_learnable_msfa.ipynb"] = notebook(
    [
        md_cell(
            """
            # Phase 3: Learnable MSFA Without Sphere-Packing Loss

            Goal: replace the fixed 4x4 assignment with a learnable tile and establish the non-regularized learned baseline.
            """
        ),
        md_cell(common_header),
        colab_mount_cell(),
        code_cell(
            """
            import csv
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, Dataset

            PROJECT_ROOT = Path("/content/drive/MyDrive/Msa-Osp")
            PATCH_PATH = PROJECT_ROOT / "dataset_patches.npz"
            HISTORY_PATH = PROJECT_ROOT / "learned_msfa_history.csv"
            CKPT_PATH = PROJECT_ROOT / "learned_msfa_best.pth"
            TILE_PATH = PROJECT_ROOT / "learned_msfa_tile_soft.npy"

            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            BATCH_SIZE = 8
            EPOCHS = 60
            UNET_LR = 2e-4
            MSFA_LR = 5e-4
            TEMP = 0.05
            BAND_COUNT = 16
            TILE_SIZE = 4
            """
        ),
        code_cell(
            """
            data = np.load(PATCH_PATH)
            train_target = data["train"]
            val_target = data["val"]

            class CubeDataset(Dataset):
                def __init__(self, cubes):
                    self.cubes = cubes

                def __len__(self):
                    return len(self.cubes)

                def __getitem__(self, idx):
                    return torch.from_numpy(self.cubes[idx]).permute(2, 0, 1).float()

            train_loader = DataLoader(CubeDataset(train_target), batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(CubeDataset(val_target), batch_size=BATCH_SIZE, shuffle=False)
            """
        ),
        code_cell(
            """
            class DoubleConv(nn.Module):
                def __init__(self, in_ch, out_ch):
                    super().__init__()
                    self.block = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_ch, out_ch, 3, padding=1),
                        nn.ReLU(inplace=True),
                    )

                def forward(self, x):
                    return self.block(x)

            class UNet2D(nn.Module):
                def __init__(self, in_ch=1, out_ch=16, base=64):
                    super().__init__()
                    self.enc1 = DoubleConv(in_ch, base)
                    self.pool1 = nn.MaxPool2d(2)
                    self.enc2 = DoubleConv(base, base * 2)
                    self.pool2 = nn.MaxPool2d(2)
                    self.bottleneck = DoubleConv(base * 2, base * 4)
                    self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
                    self.dec2 = DoubleConv(base * 4, base * 2)
                    self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
                    self.dec1 = DoubleConv(base * 2, base)
                    self.final = nn.Conv2d(base, out_ch, 1)

                def forward(self, x):
                    e1 = self.enc1(x)
                    e2 = self.enc2(self.pool1(e1))
                    b = self.bottleneck(self.pool2(e2))
                    d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
                    d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
                    return self.final(d1)

            class LearnableMSFA(nn.Module):
                def __init__(self, bands=BAND_COUNT, tile_size=TILE_SIZE, temperature=TEMP):
                    super().__init__()
                    self.tile_size = tile_size
                    self.temperature = temperature
                    self.logits = nn.Parameter(torch.randn(bands, tile_size, tile_size) * 0.01)

                def soft_tile(self):
                    return torch.softmax(self.logits / self.temperature, dim=0)

                def forward(self, x):
                    _, _, h, w = x.shape
                    weights = self.soft_tile()
                    weights_full = weights.repeat(1, h // self.tile_size, w // self.tile_size)
                    return (x * weights_full.unsqueeze(0)).sum(dim=1, keepdim=True)

            def compute_psnr(pred, target, eps=1e-8):
                mse = torch.mean((pred - target) ** 2)
                return 20 * torch.log10(1.0 / torch.sqrt(mse + eps))

            def spectral_angle_mapper(pred, target, eps=1e-8):
                dot = torch.sum(pred * target, dim=1)
                pred_norm = torch.norm(pred, dim=1)
                target_norm = torch.norm(target, dim=1)
                cos_theta = torch.clamp(dot / (pred_norm * target_norm + eps), -1 + eps, 1 - eps)
                return torch.rad2deg(torch.acos(cos_theta)).mean()
            """
        ),
        code_cell(
            """
            unet = UNet2D().to(DEVICE)
            msfa = LearnableMSFA().to(DEVICE)
            optimizer = torch.optim.Adam(
                [
                    {"params": unet.parameters(), "lr": UNET_LR},
                    {"params": msfa.parameters(), "lr": MSFA_LR},
                ]
            )
            criterion = nn.L1Loss()

            best_psnr = -float("inf")
            history = []

            for epoch in range(1, EPOCHS + 1):
                unet.train()
                msfa.train()
                train_loss = 0.0
                train_psnr = 0.0

                for x in train_loader:
                    x = x.to(DEVICE)
                    pred = unet(msfa(x))
                    loss = criterion(pred, x)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    train_psnr += compute_psnr(pred, x).item()

                unet.eval()
                msfa.eval()
                val_psnr = 0.0
                val_sam = 0.0
                with torch.no_grad():
                    for x in val_loader:
                        x = x.to(DEVICE)
                        pred = unet(msfa(x))
                        val_psnr += compute_psnr(pred, x).item()
                        val_sam += spectral_angle_mapper(pred, x).item()

                train_loss /= len(train_loader)
                train_psnr /= len(train_loader)
                val_psnr /= len(val_loader)
                val_sam /= len(val_loader)
                history.append(
                    {"epoch": epoch, "train_l1": train_loss, "train_psnr": train_psnr, "val_psnr": val_psnr, "val_sam_deg": val_sam}
                )
                print(f"Epoch {epoch:02d} | train PSNR {train_psnr:.2f} dB | val PSNR {val_psnr:.2f} dB | val SAM {val_sam:.2f} deg")

                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"epoch": epoch, "unet": unet.state_dict(), "msfa": msfa.state_dict()}, CKPT_PATH)
                    TILE_PATH.parent.mkdir(parents=True, exist_ok=True)
                    np.save(TILE_PATH, msfa.soft_tile().detach().cpu().numpy())

            HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(HISTORY_PATH, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
                writer.writeheader()
                writer.writerows(history)
            """
        ),
    ]
)

notebooks["04_phase4_sp_regularized_msfa.ipynb"] = notebook(
    [
        md_cell(
            """
            # Phase 4: Learnable MSFA With Sphere-Packing Regularization

            Goal: add a paper-aligned separation loss that discourages filter collapse in wavelength space.
            """
        ),
        md_cell(common_header),
        colab_mount_cell(),
        code_cell(
            """
            import csv
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, Dataset

            PROJECT_ROOT = Path("/content/drive/MyDrive/Msa-Osp")
            PATCH_PATH = PROJECT_ROOT / "dataset_patches.npz"
            HISTORY_PATH = PROJECT_ROOT / "learned_msfa_sp_history.csv"
            CKPT_PATH = PROJECT_ROOT / "learned_msfa_sp_best.pth"
            TILE_PATH = PROJECT_ROOT / "learned_msfa_sp_tile_soft.npy"

            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            BATCH_SIZE = 8
            EPOCHS = 60
            UNET_LR = 2e-4
            MSFA_LR = 5e-4
            TEMP = 0.05
            BAND_COUNT = 16
            TILE_SIZE = 4
            LAMBDA_SP = 5e-4
            LAMBDA_SMOOTH = 1e-4
            D_MIN = 20.0
            WAVELENGTHS = torch.linspace(400.0, 700.0, BAND_COUNT)
            """
        ),
        code_cell(
            """
            data = np.load(PATCH_PATH)
            train_target = data["train"]
            val_target = data["val"]

            class CubeDataset(Dataset):
                def __init__(self, cubes):
                    self.cubes = cubes

                def __len__(self):
                    return len(self.cubes)

                def __getitem__(self, idx):
                    return torch.from_numpy(self.cubes[idx]).permute(2, 0, 1).float()

            train_loader = DataLoader(CubeDataset(train_target), batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(CubeDataset(val_target), batch_size=BATCH_SIZE, shuffle=False)
            """
        ),
        code_cell(
            """
            class DoubleConv(nn.Module):
                def __init__(self, in_ch, out_ch):
                    super().__init__()
                    self.block = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_ch, out_ch, 3, padding=1),
                        nn.ReLU(inplace=True),
                    )

                def forward(self, x):
                    return self.block(x)

            class UNet2D(nn.Module):
                def __init__(self, in_ch=1, out_ch=16, base=64):
                    super().__init__()
                    self.enc1 = DoubleConv(in_ch, base)
                    self.pool1 = nn.MaxPool2d(2)
                    self.enc2 = DoubleConv(base, base * 2)
                    self.pool2 = nn.MaxPool2d(2)
                    self.bottleneck = DoubleConv(base * 2, base * 4)
                    self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
                    self.dec2 = DoubleConv(base * 4, base * 2)
                    self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
                    self.dec1 = DoubleConv(base * 2, base)
                    self.final = nn.Conv2d(base, out_ch, 1)

                def forward(self, x):
                    e1 = self.enc1(x)
                    e2 = self.enc2(self.pool1(e1))
                    b = self.bottleneck(self.pool2(e2))
                    d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
                    d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
                    return self.final(d1)

            class LearnableMSFA(nn.Module):
                def __init__(self, bands=BAND_COUNT, tile_size=TILE_SIZE, temperature=TEMP):
                    super().__init__()
                    self.tile_size = tile_size
                    self.temperature = temperature
                    self.logits = nn.Parameter(torch.randn(bands, tile_size, tile_size) * 0.01)

                def soft_tile(self):
                    return torch.softmax(self.logits / self.temperature, dim=0)

                def forward(self, x):
                    _, _, h, w = x.shape
                    weights = self.soft_tile()
                    weights_full = weights.repeat(1, h // self.tile_size, w // self.tile_size)
                    return (x * weights_full.unsqueeze(0)).sum(dim=1, keepdim=True)

            def compute_psnr(pred, target, eps=1e-8):
                mse = torch.mean((pred - target) ** 2)
                return 20 * torch.log10(1.0 / torch.sqrt(mse + eps))

            def spectral_angle_mapper(pred, target, eps=1e-8):
                dot = torch.sum(pred * target, dim=1)
                pred_norm = torch.norm(pred, dim=1)
                target_norm = torch.norm(target, dim=1)
                cos_theta = torch.clamp(dot / (pred_norm * target_norm + eps), -1 + eps, 1 - eps)
                return torch.rad2deg(torch.acos(cos_theta)).mean()

            def centroid_positions(msfa):
                weights = msfa.soft_tile()
                wavelengths = WAVELENGTHS.to(weights.device).view(BAND_COUNT, 1, 1)
                return (weights * wavelengths).sum(dim=0).reshape(-1)

            def pairwise_centroid_distances(msfa):
                c = centroid_positions(msfa)
                pairwise = torch.abs(c[:, None] - c[None, :])
                mask = torch.triu(torch.ones_like(pairwise), diagonal=1) > 0
                return pairwise[mask]

            def sphere_packing_loss(msfa, d_min=D_MIN):
                distances = pairwise_centroid_distances(msfa)
                penalties = torch.relu(d_min - distances)
                return penalties.mean()

            def spectral_smoothness_loss(msfa):
                weights = msfa.soft_tile()
                return torch.mean((weights[1:] - weights[:-1]) ** 2)

            def centroid_statistics(msfa):
                distances = pairwise_centroid_distances(msfa)
                return {
                    "min_centroid_distance": distances.min().item(),
                    "mean_centroid_distance": distances.mean().item(),
                }
            """
        ),
        code_cell(
            """
            unet = UNet2D().to(DEVICE)
            msfa = LearnableMSFA().to(DEVICE)
            optimizer = torch.optim.Adam(
                [
                    {"params": unet.parameters(), "lr": UNET_LR},
                    {"params": msfa.parameters(), "lr": MSFA_LR},
                ]
            )
            criterion = nn.L1Loss()
            best_psnr = -float("inf")
            history = []

            for epoch in range(1, EPOCHS + 1):
                unet.train()
                msfa.train()
                train_loss = 0.0
                train_psnr = 0.0

                for x in train_loader:
                    x = x.to(DEVICE)
                    pred = unet(msfa(x))
                    recon_loss = criterion(pred, x)
                    sp_loss = sphere_packing_loss(msfa)
                    smooth_loss = spectral_smoothness_loss(msfa)
                    loss = recon_loss + LAMBDA_SP * sp_loss + LAMBDA_SMOOTH * smooth_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    train_psnr += compute_psnr(pred, x).item()

                unet.eval()
                msfa.eval()
                val_psnr = 0.0
                val_sam = 0.0
                with torch.no_grad():
                    for x in val_loader:
                        x = x.to(DEVICE)
                        pred = unet(msfa(x))
                        val_psnr += compute_psnr(pred, x).item()
                        val_sam += spectral_angle_mapper(pred, x).item()

                train_loss /= len(train_loader)
                train_psnr /= len(train_loader)
                val_psnr /= len(val_loader)
                val_sam /= len(val_loader)
                sp_value = sphere_packing_loss(msfa).item()
                stats = centroid_statistics(msfa)
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_psnr": train_psnr,
                        "val_psnr": val_psnr,
                        "val_sam_deg": val_sam,
                        "sp_loss": sp_value,
                        "min_centroid_distance_nm": stats["min_centroid_distance"],
                        "mean_centroid_distance_nm": stats["mean_centroid_distance"],
                    }
                )
                print(
                    f"Epoch {epoch:02d} | train PSNR {train_psnr:.2f} dB | "
                    f"val PSNR {val_psnr:.2f} dB | val SAM {val_sam:.2f} deg | "
                    f"sp {sp_value:.4f} | min d {stats['min_centroid_distance']:.2f} nm"
                )

                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "unet": unet.state_dict(),
                            "msfa": msfa.state_dict(),
                            "best_val_psnr": best_psnr,
                            "lambda_sp": LAMBDA_SP,
                            "lambda_smooth": LAMBDA_SMOOTH,
                            "d_min": D_MIN,
                            "temperature": TEMP,
                            "centroid_stats": stats,
                        },
                        CKPT_PATH,
                    )
                    TILE_PATH.parent.mkdir(parents=True, exist_ok=True)
                    np.save(TILE_PATH, msfa.soft_tile().detach().cpu().numpy())

            HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(HISTORY_PATH, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
                writer.writeheader()
                writer.writerows(history)
            """
        ),
        code_cell(
            """
            checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
            msfa.load_state_dict(checkpoint["msfa"])

            soft_tile = msfa.soft_tile().detach().cpu().numpy()
            hard_tile = soft_tile.argmax(axis=0)
            centroid_nm = (soft_tile * np.linspace(400.0, 700.0, BAND_COUNT)[:, None, None]).sum(axis=0)

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(hard_tile, cmap="tab20")
            plt.title("Hard tile after SP training")
            plt.colorbar()

            plt.subplot(1, 2, 2)
            plt.imshow(centroid_nm, cmap="viridis")
            plt.title("Centroid wavelengths (nm)")
            plt.colorbar(label="nm")
            plt.tight_layout()
            plt.show()

            print("Best checkpoint centroid stats:", checkpoint.get("centroid_stats", {}))
            """
        ),
    ]
)

notebooks["05_phase5_ablations_figures_tables.ipynb"] = notebook(
    [
        md_cell(
            """
            # Phase 5: Ablations, Figures, And Tables

            Goal: aggregate finalized runs into paper-ready comparisons.
            """
        ),
        md_cell(common_header),
        colab_mount_cell(),
        code_cell(
            """
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd

            PROJECT_ROOT = Path("/content/drive/MyDrive/Msa-Osp")
            BASELINE_HISTORY = PROJECT_ROOT / "baseline_history.csv"
            LEARNED_HISTORY = PROJECT_ROOT / "learned_msfa_history.csv"
            SP_HISTORY = PROJECT_ROOT / "learned_msfa_sp_history.csv"
            BRIDGE_HISTORY = PROJECT_ROOT / "learnable_ab_history.csv"
            SP3D_HISTORY = PROJECT_ROOT / "learned_msfa_3dsp_history.csv"
            OSP_SELECTOR_REFINE_HISTORY = PROJECT_ROOT / "learned_osp_selector_refine_history.csv"
            OSP_SELECTOR_HISTORY = PROJECT_ROOT / "learned_osp_selector_history.csv"
            OSP_SEEDED_HISTORY = PROJECT_ROOT / "osp_seeded_learnable_msfa_history.csv"
            SUMMARY_PATH = PROJECT_ROOT / "paper_results_summary.csv"

            baseline = pd.read_csv(BASELINE_HISTORY)
            learned = pd.read_csv(LEARNED_HISTORY)
            learned_sp = pd.read_csv(SP_HISTORY)
            learned_ab = pd.read_csv(BRIDGE_HISTORY) if BRIDGE_HISTORY.exists() else None
            learned_sp3d = pd.read_csv(SP3D_HISTORY) if SP3D_HISTORY.exists() else None
            learned_osp_seeded = pd.read_csv(OSP_SEEDED_HISTORY) if OSP_SEEDED_HISTORY.exists() else None
            if OSP_SELECTOR_REFINE_HISTORY.exists():
                learned_osp_selector = pd.read_csv(OSP_SELECTOR_REFINE_HISTORY)
                learned_osp_selector_label = "Learned exact OSP selector + hard refine"
            elif OSP_SELECTOR_HISTORY.exists():
                learned_osp_selector = pd.read_csv(OSP_SELECTOR_HISTORY)
                learned_osp_selector_label = "Learned exact OSP selector"
            else:
                learned_osp_selector = None
                learned_osp_selector_label = None
            """
        ),
        code_cell(
            """
            rows = [
                {
                    "model": "Fixed MSFA + UNet",
                    "best_val_psnr": baseline["val_psnr"].max(),
                    "best_val_sam_deg": baseline.loc[baseline["val_psnr"].idxmax(), "val_sam_deg"],
                },
                {
                    "model": "Learned MSFA + UNet",
                    "best_val_psnr": learned["val_psnr"].max(),
                    "best_val_sam_deg": learned.loc[learned["val_psnr"].idxmax(), "val_sam_deg"],
                },
                {
                    "model": "Learned MSFA + 1D SP + UNet",
                    "best_val_psnr": learned_sp["val_psnr"].max(),
                    "best_val_sam_deg": learned_sp.loc[learned_sp["val_psnr"].idxmax(), "val_sam_deg"],
                },
            ]
            if learned_ab is not None:
                rows.append(
                    {
                        "model": "Learnable (a,b) OSP bridge + UNet",
                        "best_val_psnr": learned_ab["val_psnr"].max(),
                        "best_val_sam_deg": learned_ab.loc[learned_ab["val_psnr"].idxmax(), "val_sam_deg"],
                    }
                )
            if learned_sp3d is not None:
                rows.append(
                    {
                        "model": "Learned MSFA + 3D SP + UNet",
                        "best_val_psnr": learned_sp3d["val_psnr"].max(),
                        "best_val_sam_deg": learned_sp3d.loc[learned_sp3d["val_psnr"].idxmax(), "val_sam_deg"],
                    }
                )
            if learned_osp_seeded is not None:
                rows.append(
                    {
                        "model": "OSP-seeded learnable MSFA + 3D SP + UNet",
                        "best_val_psnr": learned_osp_seeded["val_psnr"].max(),
                        "best_val_sam_deg": learned_osp_seeded.loc[learned_osp_seeded["val_psnr"].idxmax(), "val_sam_deg"],
                    }
                )
            if learned_osp_selector is not None:
                rows.append(
                    {
                        "model": learned_osp_selector_label,
                        "best_val_psnr": learned_osp_selector["val_psnr"].max(),
                        "best_val_sam_deg": learned_osp_selector.loc[learned_osp_selector["val_psnr"].idxmax(), "val_sam_deg"],
                    }
                )

            summary = pd.DataFrame(rows)
            SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(SUMMARY_PATH, index=False)
            summary
            """
        ),
        code_cell(
            """
            plt.figure(figsize=(10, 4))
            plt.plot(baseline["epoch"], baseline["val_psnr"], label="Fixed")
            plt.plot(learned["epoch"], learned["val_psnr"], label="Learned")
            plt.plot(learned_sp["epoch"], learned_sp["val_psnr"], label="Learned+1D SP")
            if learned_ab is not None:
                plt.plot(learned_ab["epoch"], learned_ab["val_psnr"], label="Learned (a,b)")
            if learned_sp3d is not None:
                plt.plot(learned_sp3d["epoch"], learned_sp3d["val_psnr"], label="Learned+3D SP")
            if learned_osp_seeded is not None:
                plt.plot(learned_osp_seeded["epoch"], learned_osp_seeded["val_psnr"], label="OSP-seeded learnable")
            if learned_osp_selector is not None:
                plt.plot(learned_osp_selector["epoch"], learned_osp_selector["val_psnr"], label=learned_osp_selector_label)
            plt.xlabel("Epoch")
            plt.ylabel("Validation PSNR (dB)")
            plt.title("Main comparison")
            plt.legend()
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 4))
            plt.plot(baseline["epoch"], baseline["val_sam_deg"], label="Fixed")
            plt.plot(learned["epoch"], learned["val_sam_deg"], label="Learned")
            plt.plot(learned_sp["epoch"], learned_sp["val_sam_deg"], label="Learned+1D SP")
            if learned_ab is not None:
                plt.plot(learned_ab["epoch"], learned_ab["val_sam_deg"], label="Learned (a,b)")
            if learned_sp3d is not None:
                plt.plot(learned_sp3d["epoch"], learned_sp3d["val_sam_deg"], label="Learned+3D SP")
            if learned_osp_seeded is not None:
                plt.plot(learned_osp_seeded["epoch"], learned_osp_seeded["val_sam_deg"], label="OSP-seeded learnable")
            if learned_osp_selector is not None:
                plt.plot(learned_osp_selector["epoch"], learned_osp_selector["val_sam_deg"], label=learned_osp_selector_label)
            plt.xlabel("Epoch")
            plt.ylabel("Validation SAM (deg)")
            plt.title("Spectral accuracy comparison")
            plt.legend()
            plt.tight_layout()
            plt.show()
            """
        ),
    ]
)

notebooks["08_phase8_3d_osp_regularized_msfa.ipynb"] = notebook(
    [
        md_cell(
            """
            # Phase 8: Optional 3D OSP-Regularized MSFA

            Goal: upgrade the sphere-packing prior from a spectral-centroid-only surrogate to a spatio-spectral 3D surrogate.

            Practical note:
            - This notebook is optional.
            - Keep the 1D SP result as fallback.
            - Run this only after Phase 4 is already working.
            """
        ),
        md_cell(common_header),
        colab_mount_cell(),
        code_cell(
            """
            import csv
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, Dataset

            PROJECT_ROOT = Path("/content/drive/MyDrive/Msa-Osp")
            PATCH_PATH = PROJECT_ROOT / "dataset_patches.npz"
            INIT_CKPT_PATH = PROJECT_ROOT / "learned_msfa_sp_best.pth"
            HISTORY_PATH = PROJECT_ROOT / "learned_msfa_3dsp_history.csv"
            CKPT_PATH = PROJECT_ROOT / "learned_msfa_3dsp_best.pth"
            TILE_PATH = PROJECT_ROOT / "learned_msfa_3dsp_tile_soft.npy"

            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            BATCH_SIZE = 8
            EPOCHS = 30
            UNET_LR = 1e-4
            MSFA_LR = 2e-4
            TEMP = 0.05
            BAND_COUNT = 16
            TILE_SIZE = 4
            LAMBDA_SP = 5e-4
            LAMBDA_SMOOTH = 1e-4
            D_MIN_3D = 0.55
            Z_WEIGHT = 1.25
            WAVELENGTHS = torch.linspace(0.0, 1.0, BAND_COUNT)

            print("Warm-start checkpoint exists:", INIT_CKPT_PATH.exists())
            """
        ),
        code_cell(
            """
            data = np.load(PATCH_PATH)
            train_target = data["train"]
            val_target = data["val"]

            class CubeDataset(Dataset):
                def __init__(self, cubes):
                    self.cubes = cubes

                def __len__(self):
                    return len(self.cubes)

                def __getitem__(self, idx):
                    return torch.from_numpy(self.cubes[idx]).permute(2, 0, 1).float()

            train_loader = DataLoader(CubeDataset(train_target), batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(CubeDataset(val_target), batch_size=BATCH_SIZE, shuffle=False)
            """
        ),
        code_cell(
            """
            class DoubleConv(nn.Module):
                def __init__(self, in_ch, out_ch):
                    super().__init__()
                    self.block = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_ch, out_ch, 3, padding=1),
                        nn.ReLU(inplace=True),
                    )

                def forward(self, x):
                    return self.block(x)

            class UNet2D(nn.Module):
                def __init__(self, in_ch=1, out_ch=16, base=64):
                    super().__init__()
                    self.enc1 = DoubleConv(in_ch, base)
                    self.pool1 = nn.MaxPool2d(2)
                    self.enc2 = DoubleConv(base, base * 2)
                    self.pool2 = nn.MaxPool2d(2)
                    self.bottleneck = DoubleConv(base * 2, base * 4)
                    self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
                    self.dec2 = DoubleConv(base * 4, base * 2)
                    self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
                    self.dec1 = DoubleConv(base * 2, base)
                    self.final = nn.Conv2d(base, out_ch, 1)

                def forward(self, x):
                    e1 = self.enc1(x)
                    e2 = self.enc2(self.pool1(e1))
                    b = self.bottleneck(self.pool2(e2))
                    d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
                    d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
                    return self.final(d1)

            class LearnableMSFA(nn.Module):
                def __init__(self, bands=BAND_COUNT, tile_size=TILE_SIZE, temperature=TEMP):
                    super().__init__()
                    self.tile_size = tile_size
                    self.temperature = temperature
                    self.logits = nn.Parameter(torch.randn(bands, tile_size, tile_size) * 0.01)

                def soft_tile(self):
                    return torch.softmax(self.logits / self.temperature, dim=0)

                def forward(self, x):
                    _, _, h, w = x.shape
                    weights = self.soft_tile()
                    weights_full = weights.repeat(1, h // self.tile_size, w // self.tile_size)
                    return (x * weights_full.unsqueeze(0)).sum(dim=1, keepdim=True)

            def compute_psnr(pred, target, eps=1e-8):
                mse = torch.mean((pred - target) ** 2)
                return 20 * torch.log10(1.0 / torch.sqrt(mse + eps))

            def spectral_angle_mapper(pred, target, eps=1e-8):
                dot = torch.sum(pred * target, dim=1)
                pred_norm = torch.norm(pred, dim=1)
                target_norm = torch.norm(target, dim=1)
                cos_theta = torch.clamp(dot / (pred_norm * target_norm + eps), -1 + eps, 1 - eps)
                return torch.rad2deg(torch.acos(cos_theta)).mean()

            def spectral_smoothness_loss(msfa):
                weights = msfa.soft_tile()
                return torch.mean((weights[1:] - weights[:-1]) ** 2)
            """
        ),
        code_cell(
            """
            spatial_coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(0.0, 1.0, TILE_SIZE),
                    torch.linspace(0.0, 1.0, TILE_SIZE),
                    indexing="ij",
                ),
                dim=-1,
            ).reshape(-1, 2)

            def spatio_spectral_points(msfa):
                weights = msfa.soft_tile()
                z = (weights * WAVELENGTHS.to(weights.device).view(BAND_COUNT, 1, 1)).sum(dim=0).reshape(-1, 1)
                xy = spatial_coords.to(weights.device)
                return torch.cat([xy, Z_WEIGHT * z], dim=1)

            def pairwise_3d_distances(msfa):
                points = spatio_spectral_points(msfa)
                d = torch.cdist(points, points, p=2)
                mask = torch.triu(torch.ones_like(d), diagonal=1) > 0
                return d[mask]

            def osp_3d_loss(msfa, d_min=D_MIN_3D):
                distances = pairwise_3d_distances(msfa)
                return torch.relu(d_min - distances).mean()

            def distance_stats(msfa):
                distances = pairwise_3d_distances(msfa)
                return {
                    "min_3d_distance": distances.min().item(),
                    "mean_3d_distance": distances.mean().item(),
                }
            """
        ),
        code_cell(
            """
            unet = UNet2D().to(DEVICE)
            msfa = LearnableMSFA().to(DEVICE)

            if INIT_CKPT_PATH.exists():
                checkpoint = torch.load(INIT_CKPT_PATH, map_location=DEVICE)
                unet.load_state_dict(checkpoint["unet"])
                msfa.load_state_dict(checkpoint["msfa"])
                print("Warm-started from 1D SP checkpoint.")
            else:
                print("Starting from scratch.")

            optimizer = torch.optim.Adam(
                [
                    {"params": unet.parameters(), "lr": UNET_LR},
                    {"params": msfa.parameters(), "lr": MSFA_LR},
                ]
            )
            criterion = nn.L1Loss()
            best_psnr = -float("inf")
            history = []

            for epoch in range(1, EPOCHS + 1):
                unet.train()
                msfa.train()
                train_loss = 0.0
                train_psnr = 0.0

                for x in train_loader:
                    x = x.to(DEVICE)
                    pred = unet(msfa(x))
                    recon_loss = criterion(pred, x)
                    sp_loss = osp_3d_loss(msfa)
                    smooth_loss = spectral_smoothness_loss(msfa)
                    loss = recon_loss + LAMBDA_SP * sp_loss + LAMBDA_SMOOTH * smooth_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    train_psnr += compute_psnr(pred, x).item()

                unet.eval()
                msfa.eval()
                val_psnr = 0.0
                val_sam = 0.0
                with torch.no_grad():
                    for x in val_loader:
                        x = x.to(DEVICE)
                        pred = unet(msfa(x))
                        val_psnr += compute_psnr(pred, x).item()
                        val_sam += spectral_angle_mapper(pred, x).item()

                train_loss /= len(train_loader)
                train_psnr /= len(train_loader)
                val_psnr /= len(val_loader)
                val_sam /= len(val_loader)
                sp_value = osp_3d_loss(msfa).item()
                stats = distance_stats(msfa)
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_psnr": train_psnr,
                        "val_psnr": val_psnr,
                        "val_sam_deg": val_sam,
                        "osp_3d_loss": sp_value,
                        "min_3d_distance": stats["min_3d_distance"],
                        "mean_3d_distance": stats["mean_3d_distance"],
                    }
                )
                print(
                    f"Epoch {epoch:02d} | train PSNR {train_psnr:.2f} dB | "
                    f"val PSNR {val_psnr:.2f} dB | val SAM {val_sam:.2f} deg | "
                    f"3D sp {sp_value:.4f} | min 3D d {stats['min_3d_distance']:.3f}"
                )

                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "unet": unet.state_dict(),
                            "msfa": msfa.state_dict(),
                            "best_val_psnr": best_psnr,
                            "lambda_sp": LAMBDA_SP,
                            "lambda_smooth": LAMBDA_SMOOTH,
                            "d_min_3d": D_MIN_3D,
                            "z_weight": Z_WEIGHT,
                            "temperature": TEMP,
                            "distance_stats": stats,
                        },
                        CKPT_PATH,
                    )
                    TILE_PATH.parent.mkdir(parents=True, exist_ok=True)
                    np.save(TILE_PATH, msfa.soft_tile().detach().cpu().numpy())

            HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(HISTORY_PATH, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
                writer.writeheader()
                writer.writerows(history)
            """
        ),
        code_cell(
            """
            checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
            msfa.load_state_dict(checkpoint["msfa"])

            soft_tile = msfa.soft_tile().detach().cpu().numpy()
            hard_tile = soft_tile.argmax(axis=0)
            centroid_nm = (soft_tile * np.linspace(400.0, 700.0, BAND_COUNT)[:, None, None]).sum(axis=0)

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(hard_tile, cmap="tab20")
            plt.title("Hard tile after 3D SP training")
            plt.colorbar()

            plt.subplot(1, 2, 2)
            plt.imshow(centroid_nm, cmap="viridis")
            plt.title("Centroid wavelengths (nm)")
            plt.colorbar(label="nm")
            plt.tight_layout()
            plt.show()

            print("Best 3D distance stats:", checkpoint.get("distance_stats", {}))
            """
        ),
        md_cell(
            """
            If this 3D variant does not clearly beat the 1D SP result, keep the 1D SP result as the main paper line and report the 3D version as an exploratory extension.
            """
        ),
    ]
)

notebooks["09_phase9_learnable_ab_bridge.ipynb"] = notebook(
    [
        md_cell(
            """
            # Phase 9: Learnable `(a,b)` OSP Bridge

            Goal: build a constrained learnable bridge back to the original OSP generator family.

            Pattern family:
            `G = mod(I * a + J * b, NF) + 1`

            This notebook does not replace the full learned MSFA model.
            It is the OSP-faithful bridge experiment.
            """
        ),
        md_cell(common_header),
        colab_mount_cell(),
        code_cell(
            """
            import csv
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, Dataset

            PROJECT_ROOT = Path("/content/drive/MyDrive/Msa-Osp")
            PATCH_PATH = PROJECT_ROOT / "dataset_patches.npz"
            HISTORY_PATH = PROJECT_ROOT / "learnable_ab_history.csv"
            CKPT_PATH = PROJECT_ROOT / "learnable_ab_best.pth"
            FIG_PATH = PROJECT_ROOT / "learnable_ab_pattern.png"

            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            BATCH_SIZE = 8
            LR = 2e-4
            BAND_COUNT = 16
            TILE_SIZE = 16
            A_MAX = BAND_COUNT / 2
            B_MAX = BAND_COUNT / 2
            EPOCHS = 80
            SHARPNESS_START = 8.0
            SHARPNESS_END = 20.0
            LAMBDA_OSP = 1e-3
            LAMBDA_UNIFORMITY = 5e-3
            LAMBDA_ENTROPY = 1e-3
            D_MIN_3D = 0.55
            Z_WEIGHT = 1.25

            print("Device:", DEVICE)
            """
        ),
        code_cell(
            """
            data = np.load(PATCH_PATH)
            train_target = data["train"]
            val_target = data["val"]

            class CubeDataset(Dataset):
                def __init__(self, cubes):
                    self.cubes = cubes

                def __len__(self):
                    return len(self.cubes)

                def __getitem__(self, idx):
                    return torch.from_numpy(self.cubes[idx]).permute(2, 0, 1).float()

            train_loader = DataLoader(CubeDataset(train_target), batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(CubeDataset(val_target), batch_size=BATCH_SIZE, shuffle=False)
            """
        ),
        code_cell(
            """
            class DoubleConv(nn.Module):
                def __init__(self, in_ch, out_ch):
                    super().__init__()
                    self.block = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_ch, out_ch, 3, padding=1),
                        nn.ReLU(inplace=True),
                    )

                def forward(self, x):
                    return self.block(x)

            class UNet2D(nn.Module):
                def __init__(self, in_ch=16, out_ch=16, base=64):
                    super().__init__()
                    self.enc1 = DoubleConv(in_ch, base)
                    self.pool1 = nn.MaxPool2d(2)
                    self.enc2 = DoubleConv(base, base * 2)
                    self.pool2 = nn.MaxPool2d(2)
                    self.bottleneck = DoubleConv(base * 2, base * 4)
                    self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
                    self.dec2 = DoubleConv(base * 4, base * 2)
                    self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
                    self.dec1 = DoubleConv(base * 2, base)
                    self.final = nn.Conv2d(base, out_ch, 1)

                def forward(self, x):
                    e1 = self.enc1(x)
                    e2 = self.enc2(self.pool1(e1))
                    b = self.bottleneck(self.pool2(e2))
                    d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
                    d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
                    return self.final(d1)

            class LearnableABMask(nn.Module):
                def __init__(self, bands=BAND_COUNT, tile_size=TILE_SIZE):
                    super().__init__()
                    self.bands = bands
                    self.tile_size = tile_size
                    self.raw_a = nn.Parameter(torch.tensor(0.0))
                    self.raw_b = nn.Parameter(torch.tensor(0.0))

                    i = torch.arange(1, tile_size + 1, dtype=torch.float32)
                    j = torch.arange(1, tile_size + 1, dtype=torch.float32)
                    I, J = torch.meshgrid(i, j, indexing="ij")
                    self.register_buffer("I", I)
                    self.register_buffer("J", J)
                    self.register_buffer("band_ids", torch.arange(1, bands + 1, dtype=torch.float32).view(1, 1, bands))

                def continuous_ab(self):
                    a = 1.0 + (A_MAX - 1.0) * torch.sigmoid(self.raw_a)
                    b = 1.0 + (B_MAX - 1.0) * torch.sigmoid(self.raw_b)
                    return a, b

                def soft_tile(self, sharpness=SHARPNESS_START):
                    a, b = self.continuous_ab()
                    value = torch.remainder(self.I * a + self.J * b - 1.0, float(self.bands)) + 1.0
                    dist = torch.abs(value.unsqueeze(-1) - self.band_ids)
                    circular = torch.minimum(dist, float(self.bands) - dist)
                    logits = -sharpness * circular
                    weights = torch.softmax(logits, dim=-1)
                    return weights.permute(2, 0, 1)

                def hard_tile(self, sharpness=SHARPNESS_END):
                    return self.soft_tile(sharpness=sharpness).argmax(dim=0) + 1

                def forward(self, x, sharpness=SHARPNESS_START):
                    _, _, h, w = x.shape
                    tile = self.soft_tile(sharpness=sharpness)
                    tile_full = tile.repeat(1, h // self.tile_size, w // self.tile_size)
                    return x * tile_full.unsqueeze(0)

            def compute_psnr(pred, target, eps=1e-8):
                mse = torch.mean((pred - target) ** 2)
                return 20 * torch.log10(1.0 / torch.sqrt(mse + eps))

            def spectral_angle_mapper(pred, target, eps=1e-8):
                dot = torch.sum(pred * target, dim=1)
                pred_norm = torch.norm(pred, dim=1)
                target_norm = torch.norm(target, dim=1)
                cos_theta = torch.clamp(dot / (pred_norm * target_norm + eps), -1 + eps, 1 - eps)
                return torch.rad2deg(torch.acos(cos_theta)).mean()

            spatial_coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(0.0, 1.0, TILE_SIZE),
                    torch.linspace(0.0, 1.0, TILE_SIZE),
                    indexing="ij",
                ),
                dim=-1,
            ).reshape(-1, 2)

            wavelength_coords = torch.linspace(0.0, 1.0, BAND_COUNT)

            def bridge_points(mask_model, sharpness=SHARPNESS_START):
                soft_tile = mask_model.soft_tile(sharpness=sharpness)
                z = (soft_tile * wavelength_coords.to(soft_tile.device).view(BAND_COUNT, 1, 1)).sum(dim=0).reshape(-1, 1)
                xy = spatial_coords.to(soft_tile.device)
                return torch.cat([xy, Z_WEIGHT * z], dim=1)

            def bridge_osp_loss(mask_model, d_min=D_MIN_3D, sharpness=SHARPNESS_START):
                points = bridge_points(mask_model, sharpness=sharpness)
                d = torch.cdist(points, points, p=2)
                mask = torch.triu(torch.ones_like(d), diagonal=1) > 0
                distances = d[mask]
                return torch.relu(d_min - distances).mean()

            def bridge_distance_stats(mask_model, sharpness=SHARPNESS_END):
                points = bridge_points(mask_model, sharpness=sharpness)
                d = torch.cdist(points, points, p=2)
                mask = torch.triu(torch.ones_like(d), diagonal=1) > 0
                distances = d[mask]
                return {
                    "min_3d_distance": distances.min().item(),
                    "mean_3d_distance": distances.mean().item(),
                }

            def uniformity_loss(mask_model, sharpness=SHARPNESS_START):
                soft_tile = mask_model.soft_tile(sharpness=sharpness)
                usage = soft_tile.mean(dim=(1, 2))
                target = torch.full_like(usage, 1.0 / BAND_COUNT)
                return torch.mean((usage - target) ** 2)

            def entropy_loss(mask_model, sharpness=SHARPNESS_START, eps=1e-8):
                soft_tile = mask_model.soft_tile(sharpness=sharpness)
                entropy = -(soft_tile * torch.log(soft_tile + eps)).sum(dim=0)
                return entropy.mean()
            """
        ),
        code_cell(
            """
            model = UNet2D().to(DEVICE)
            mask_model = LearnableABMask().to(DEVICE)
            optimizer = torch.optim.Adam(
                [
                    {"params": model.parameters(), "lr": LR},
                    {"params": mask_model.parameters(), "lr": LR},
                ]
            )
            criterion = nn.L1Loss()
            best_psnr = -float("inf")
            history = []

            for epoch in range(1, EPOCHS + 1):
                model.train()
                mask_model.train()
                train_loss = 0.0
                train_psnr = 0.0
                sharpness = SHARPNESS_START + (SHARPNESS_END - SHARPNESS_START) * (epoch - 1) / max(EPOCHS - 1, 1)

                for x in train_loader:
                    x = x.to(DEVICE)
                    sensed = mask_model(x, sharpness=sharpness)
                    pred = model(sensed)
                    recon_loss = criterion(pred, x)
                    osp_loss = bridge_osp_loss(mask_model, sharpness=sharpness)
                    uniform_loss = uniformity_loss(mask_model, sharpness=sharpness)
                    discrete_loss = entropy_loss(mask_model, sharpness=sharpness)
                    loss = (
                        recon_loss
                        + LAMBDA_OSP * osp_loss
                        + LAMBDA_UNIFORMITY * uniform_loss
                        + LAMBDA_ENTROPY * discrete_loss
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    train_psnr += compute_psnr(pred, x).item()

                model.eval()
                mask_model.eval()
                val_psnr = 0.0
                val_sam = 0.0
                with torch.no_grad():
                    for x in val_loader:
                        x = x.to(DEVICE)
                        pred = model(mask_model(x, sharpness=sharpness))
                        val_psnr += compute_psnr(pred, x).item()
                        val_sam += spectral_angle_mapper(pred, x).item()

                train_loss /= len(train_loader)
                train_psnr /= len(train_loader)
                val_psnr /= len(val_loader)
                val_sam /= len(val_loader)
                a, b = mask_model.continuous_ab()
                stats = bridge_distance_stats(mask_model, sharpness=sharpness)
                osp_value = bridge_osp_loss(mask_model, sharpness=sharpness).item()
                uniform_value = uniformity_loss(mask_model, sharpness=sharpness).item()
                discrete_value = entropy_loss(mask_model, sharpness=sharpness).item()
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_psnr": train_psnr,
                        "val_psnr": val_psnr,
                        "val_sam_deg": val_sam,
                        "a_cont": a.item(),
                        "b_cont": b.item(),
                        "sharpness": sharpness,
                        "bridge_osp_loss": osp_value,
                        "uniformity_loss": uniform_value,
                        "entropy_loss": discrete_value,
                        "min_3d_distance": stats["min_3d_distance"],
                        "mean_3d_distance": stats["mean_3d_distance"],
                    }
                )
                print(
                    f"Epoch {epoch:02d} | train PSNR {train_psnr:.2f} dB | "
                    f"val PSNR {val_psnr:.2f} dB | val SAM {val_sam:.2f} deg | "
                    f"a {a.item():.3f} | b {b.item():.3f} | "
                    f"osp {osp_value:.4f} | uni {uniform_value:.4f} | ent {discrete_value:.4f} | "
                    f"min 3D d {stats['min_3d_distance']:.3f} | sharp {sharpness:.1f}"
                )

                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model": model.state_dict(),
                            "mask_model": mask_model.state_dict(),
                            "best_val_psnr": best_psnr,
                            "a_cont": a.item(),
                            "b_cont": b.item(),
                            "sharpness": sharpness,
                            "lambda_osp": LAMBDA_OSP,
                            "lambda_uniformity": LAMBDA_UNIFORMITY,
                            "lambda_entropy": LAMBDA_ENTROPY,
                            "d_min_3d": D_MIN_3D,
                            "z_weight": Z_WEIGHT,
                            "distance_stats": stats,
                            "hard_tile": mask_model.hard_tile(sharpness=sharpness).detach().cpu().numpy(),
                        },
                        CKPT_PATH,
                    )

            HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(HISTORY_PATH, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
                writer.writeheader()
                writer.writerows(history)
            """
        ),
        code_cell(
            """
            checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
            mask_model.load_state_dict(checkpoint["mask_model"])
            final_sharpness = checkpoint.get("sharpness", SHARPNESS_END)
            hard_tile = mask_model.hard_tile(sharpness=final_sharpness).detach().cpu().numpy()
            soft_tile = mask_model.soft_tile(sharpness=final_sharpness).detach().cpu().numpy()

            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(hard_tile, cmap="tab20")
            plt.title(f"Learnable (a,b) hard tile\\na={checkpoint['a_cont']:.3f}, b={checkpoint['b_cont']:.3f}")
            plt.colorbar()

            plt.subplot(1, 2, 2)
            for i in range(soft_tile.shape[1]):
                for j in range(soft_tile.shape[2]):
                    plt.plot(soft_tile[:, i, j], alpha=0.7)
            plt.title("Soft assignments induced by learnable (a,b)")
            plt.xlabel("Band")
            plt.ylabel("Weight")
            plt.tight_layout()
            FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
            plt.show()

            print("Best checkpoint sharpness:", final_sharpness)
            print("Best checkpoint distance stats:", checkpoint.get("distance_stats", {}))
            """
        ),
        md_cell(
            """
            ## How to use this notebook

            Use it as the bridge experiment:
            - `Fixed OSP`
            - `Learnable (a,b)` OSP-family bridge
            - `Learned MSFA + 3D SP`

            If this bridge underperforms the full learned model, that strengthens the argument for moving beyond the original constrained OSP family.
            """
        ),
    ]
)

notebooks["10_phase10_discrete_osp_selector.ipynb"] = notebook(
    [
        md_cell(
            """
            # Phase 10: Learned Discrete OSP Candidate Selector

            Goal: learn over the exact discrete OSP candidate family instead of a continuous surrogate.

            This notebook keeps the final filter exact:
            - generate the valid `(a,b)` candidates from the OSP search space
            - learn a small selector over those candidates
            - jointly train the selector with a lightweight reconstructor
            - anneal and briefly reheat the selector temperature to avoid poor local minima
            """
        ),
        md_cell(common_header),
        colab_mount_cell(),
        code_cell(
            """
            import csv
            import math
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch.utils.data import DataLoader, Dataset

            PROJECT_ROOT = Path("/content/drive/MyDrive/Msa-Osp")
            PATCH_PATH = PROJECT_ROOT / "dataset_patches.npz"
            BASELINE_CKPT_PATH = PROJECT_ROOT / "unet_baseline_best.pth"
            HISTORY_PATH = PROJECT_ROOT / "learned_osp_selector_history.csv"
            CKPT_PATH = PROJECT_ROOT / "learned_osp_selector_best.pth"
            SWEEP_HISTORY_PATH = PROJECT_ROOT / "learned_osp_selector_hard_sweep.csv"
            REFINE_HISTORY_PATH = PROJECT_ROOT / "learned_osp_selector_refine_history.csv"
            REFINE_CKPT_PATH = PROJECT_ROOT / "learned_osp_selector_refine_best.pth"
            FIG_PATH = PROJECT_ROOT / "learned_osp_selector_pattern.png"

            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            BATCH_SIZE = 8
            SELECTOR_EPOCHS = 40
            HARD_SWEEP_EPOCHS = 8
            FINAL_REFINE_EPOCHS = 16
            MODEL_LR = 1e-4
            SELECTOR_LR = 2e-2
            REFINE_LR = 5e-5
            BAND_COUNT = 16
            TILE_SIZE = 16
            BASE = 32
            TEMP_START = 2.5
            TEMP_END = 0.25
            REHEAT_START = 24
            REHEAT_LENGTH = 6
            REHEAT_TEMP = 1.20
            LAMBDA_SCORE = 5e-2
            LAMBDA_SELECTOR_ENT = 2e-3
            TRAIN_SAM_WEIGHT = 5e-2
            USE_GUMBEL = True

            print("Device:", DEVICE)
            print("Warm-start baseline exists:", BASELINE_CKPT_PATH.exists())
            print("Notebook goal: exact OSP candidate selection, then hard-filter fine-tuning.")
            """
        ),
        code_cell(
            """
            data = np.load(PATCH_PATH)
            train_target = data["train"]
            val_target = data["val"]

            class CubeDataset(Dataset):
                def __init__(self, cubes):
                    self.cubes = cubes

                def __len__(self):
                    return len(self.cubes)

                def __getitem__(self, idx):
                    return torch.from_numpy(self.cubes[idx]).permute(2, 0, 1).float()

            train_loader = DataLoader(CubeDataset(train_target), batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(CubeDataset(val_target), batch_size=BATCH_SIZE, shuffle=False)
            """
        ),
        code_cell(
            """
            class DoubleConv(nn.Module):
                def __init__(self, in_ch, out_ch):
                    super().__init__()
                    self.block = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_ch, out_ch, 3, padding=1),
                        nn.ReLU(inplace=True),
                    )

                def forward(self, x):
                    return self.block(x)

            class UNet2D(nn.Module):
                def __init__(self, in_ch=16, out_ch=16, base=BASE):
                    super().__init__()
                    self.enc1 = DoubleConv(in_ch, base)
                    self.pool1 = nn.MaxPool2d(2)
                    self.enc2 = DoubleConv(base, base * 2)
                    self.pool2 = nn.MaxPool2d(2)
                    self.bottleneck = DoubleConv(base * 2, base * 4)
                    self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
                    self.dec2 = DoubleConv(base * 4, base * 2)
                    self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
                    self.dec1 = DoubleConv(base * 2, base)
                    self.final = nn.Conv2d(base, out_ch, 1)

                def forward(self, x):
                    e1 = self.enc1(x)
                    e2 = self.enc2(self.pool1(e1))
                    b = self.bottleneck(self.pool2(e2))
                    d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
                    d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
                    return self.final(d1)

            def compute_psnr(pred, target, eps=1e-8):
                mse = torch.mean((pred - target) ** 2)
                return 20 * torch.log10(1.0 / torch.sqrt(mse + eps))

            def spectral_angle_mapper(pred, target, eps=1e-8):
                dot = torch.sum(pred * target, dim=1)
                pred_norm = torch.norm(pred, dim=1)
                target_norm = torch.norm(target, dim=1)
                cos_theta = torch.clamp(dot / (pred_norm * target_norm + eps), -1 + eps, 1 - eps)
                return torch.rad2deg(torch.acos(cos_theta)).mean()

            def spectral_angle_loss(pred, target, eps=1e-8):
                dot = torch.sum(pred * target, dim=1)
                pred_norm = torch.norm(pred, dim=1)
                target_norm = torch.norm(target, dim=1)
                cos_theta = torch.clamp(dot / (pred_norm * target_norm + eps), -1 + eps, 1 - eps)
                return torch.acos(cos_theta).mean()

            def make_candidate_tile(a, b):
                idx = torch.arange(1, BAND_COUNT + 1, dtype=torch.int64)
                I, J = torch.meshgrid(idx, idx, indexing="ij")
                return torch.remainder(I * a + J * b, BAND_COUNT) + 1

            def is_uniform_candidate(tile):
                target = set(range(1, BAND_COUNT + 1))
                row_ok = all(set(tile[r, :].tolist()) == target for r in range(TILE_SIZE))
                col_ok = all(set(tile[:, c].tolist()) == target for c in range(TILE_SIZE))
                return row_ok and col_ok

            def candidate_score(tile):
                idx = torch.arange(1, TILE_SIZE + 1, dtype=torch.float32)
                I, J = torch.meshgrid(idx, idx, indexing="ij")
                points = torch.stack([I.reshape(-1), J.reshape(-1), tile.float().reshape(-1)], dim=1)
                return torch.pdist(points).min().item()

            def tile_to_mask(tile):
                return F.one_hot((tile - 1).long(), num_classes=BAND_COUNT).permute(2, 0, 1).float()

            def apply_candidate_mask(x, mask):
                _, _, h, w = x.shape
                tile_full = mask.repeat(1, h // TILE_SIZE, w // TILE_SIZE)
                return x * tile_full.unsqueeze(0)

            candidate_records = []
            for a in range(1, BAND_COUNT // 2 + 1):
                for b in range(a, BAND_COUNT // 2 + 1):
                    if math.gcd(a, BAND_COUNT) != 1 or math.gcd(b, BAND_COUNT) != 1:
                        continue
                    tile = make_candidate_tile(a, b)
                    if len(torch.unique(tile)) != BAND_COUNT:
                        continue
                    if not is_uniform_candidate(tile):
                        continue
                    candidate_records.append(
                        {
                            "a": a,
                            "b": b,
                            "tile": tile,
                            "score_raw": candidate_score(tile),
                        }
                    )

            candidate_tiles = torch.stack([tile_to_mask(rec["tile"]) for rec in candidate_records], dim=0)
            candidate_scores_raw = torch.tensor([rec["score_raw"] for rec in candidate_records], dtype=torch.float32)
            candidate_scores_norm = (candidate_scores_raw - candidate_scores_raw.min()) / (
                candidate_scores_raw.max() - candidate_scores_raw.min() + 1e-8
            )
            candidate_ab = torch.tensor([[rec["a"], rec["b"]] for rec in candidate_records], dtype=torch.int64)

            print("Discrete OSP candidate count:", len(candidate_records))
            print("Candidate (a,b) pairs:", [tuple(map(int, ab)) for ab in candidate_ab.tolist()])
            print("Candidate raw scores:", candidate_scores_raw.tolist())

            class OSPCandidateSelector(nn.Module):
                def __init__(self, candidate_tiles, candidate_scores_raw, candidate_scores_norm, candidate_ab):
                    super().__init__()
                    self.register_buffer("candidate_tiles", candidate_tiles)
                    self.register_buffer("candidate_scores_raw", candidate_scores_raw)
                    self.register_buffer("candidate_scores_norm", candidate_scores_norm)
                    self.register_buffer("candidate_ab", candidate_ab)
                    self.logits = nn.Parameter(0.25 * candidate_scores_norm.clone())

                def probs(self, temp, training=True):
                    if training and USE_GUMBEL:
                        return F.gumbel_softmax(self.logits, tau=temp, hard=False, dim=0)
                    return torch.softmax(self.logits / temp, dim=0)

                def soft_tile(self, temp, training=True):
                    probs = self.probs(temp, training=training)
                    return torch.sum(probs.view(-1, 1, 1, 1) * self.candidate_tiles, dim=0)

                def expected_score(self, temp, training=True):
                    probs = self.probs(temp, training=training)
                    return torch.sum(probs * self.candidate_scores_norm)

                def selector_entropy(self, temp, training=True, eps=1e-8):
                    probs = self.probs(temp, training=training)
                    return -(probs * torch.log(probs + eps)).sum()

                def hard_index(self):
                    return int(torch.argmax(self.logits).item())

                def hard_tile(self):
                    idx = self.hard_index()
                    return self.candidate_tiles[idx].argmax(dim=0) + 1

                def selected_ab(self):
                    idx = self.hard_index()
                    ab = self.candidate_ab[idx].tolist()
                    return idx, (int(ab[0]), int(ab[1]))

                def selected_raw_score(self):
                    return float(self.candidate_scores_raw[self.hard_index()].item())

                def forward(self, x, temp, training=True):
                    _, _, h, w = x.shape
                    tile = self.soft_tile(temp=temp, training=training)
                    tile_full = tile.repeat(1, h // TILE_SIZE, w // TILE_SIZE)
                    return x * tile_full.unsqueeze(0)
            """
        ),
        code_cell(
            """
            def selector_temperature(epoch):
                base = TEMP_START + (TEMP_END - TEMP_START) * (epoch - 1) / max(SELECTOR_EPOCHS - 1, 1)
                if REHEAT_START <= epoch < REHEAT_START + REHEAT_LENGTH:
                    frac = (epoch - REHEAT_START) / max(REHEAT_LENGTH - 1, 1)
                    return REHEAT_TEMP + frac * (base - REHEAT_TEMP)
                return base

            model = UNet2D().to(DEVICE)
            selector = OSPCandidateSelector(
                candidate_tiles=candidate_tiles,
                candidate_scores_raw=candidate_scores_raw,
                candidate_scores_norm=candidate_scores_norm,
                candidate_ab=candidate_ab,
            ).to(DEVICE)

            if BASELINE_CKPT_PATH.exists():
                checkpoint = torch.load(BASELINE_CKPT_PATH, map_location=DEVICE, weights_only=False)
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                print("Warm-started reconstructor from fixed-MSFA baseline.")
            else:
                print("Starting reconstructor from scratch.")

            optimizer = torch.optim.Adam(
                [
                    {"params": model.parameters(), "lr": MODEL_LR},
                    {"params": selector.parameters(), "lr": SELECTOR_LR},
                ]
            )
            criterion = nn.L1Loss()
            best_psnr = -float("inf")
            history = []

            for epoch in range(1, SELECTOR_EPOCHS + 1):
                temp = selector_temperature(epoch)
                model.train()
                selector.train()
                train_loss = 0.0
                train_psnr = 0.0

                for x in train_loader:
                    x = x.to(DEVICE)
                    sensed = selector(x, temp=temp, training=True)
                    pred = model(sensed)
                    recon_loss = criterion(pred, x)
                    score_loss = -selector.expected_score(temp=temp, training=True)
                    selector_ent = selector.selector_entropy(temp=temp, training=True)
                    loss = recon_loss + LAMBDA_SCORE * score_loss + LAMBDA_SELECTOR_ENT * selector_ent
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    train_psnr += compute_psnr(pred, x).item()

                model.eval()
                selector.eval()
                val_psnr = 0.0
                val_sam = 0.0
                with torch.no_grad():
                    for x in val_loader:
                        x = x.to(DEVICE)
                        pred = model(selector(x, temp=temp, training=False))
                        val_psnr += compute_psnr(pred, x).item()
                        val_sam += spectral_angle_mapper(pred, x).item()

                train_loss /= len(train_loader)
                train_psnr /= len(train_loader)
                val_psnr /= len(val_loader)
                val_sam /= len(val_loader)
                probs = selector.probs(temp=temp, training=False).detach().cpu()
                selected_idx, (selected_a, selected_b) = selector.selected_ab()
                selected_score_raw = selector.selected_raw_score()
                expected_score_norm = selector.expected_score(temp=temp, training=False).item()
                selector_entropy = selector.selector_entropy(temp=temp, training=False).item()
                top_prob = probs.max().item()
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_psnr": train_psnr,
                        "val_psnr": val_psnr,
                        "val_sam_deg": val_sam,
                        "temperature": temp,
                        "selected_index": selected_idx,
                        "selected_a": selected_a,
                        "selected_b": selected_b,
                        "selected_score_raw": selected_score_raw,
                        "expected_score_norm": expected_score_norm,
                        "selector_entropy": selector_entropy,
                        "top_probability": top_prob,
                    }
                )
                print(
                    f"Epoch {epoch:02d} | train PSNR {train_psnr:.2f} dB | "
                    f"val PSNR {val_psnr:.2f} dB | val SAM {val_sam:.2f} deg | "
                    f"temp {temp:.2f} | sel ({selected_a},{selected_b}) | "
                    f"score {selected_score_raw:.3f} | top p {top_prob:.3f}"
                )

                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model": model.state_dict(),
                            "selector": selector.state_dict(),
                            "best_val_psnr": best_psnr,
                            "temperature": temp,
                            "selected_index": selected_idx,
                            "selected_a": selected_a,
                            "selected_b": selected_b,
                            "selected_score_raw": selected_score_raw,
                            "candidate_ab": candidate_ab.cpu().numpy(),
                            "candidate_scores_raw": candidate_scores_raw.cpu().numpy(),
                            "candidate_probs": probs.numpy(),
                            "hard_tile": selector.hard_tile().detach().cpu().numpy(),
                        },
                        CKPT_PATH,
                    )

            HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(HISTORY_PATH, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
                writer.writeheader()
                writer.writerows(history)

            # Stage 2: sweep over exact geometry-optimal OSP candidates under hard training, then fine-tune the winner.
            stage1_checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
            selector.load_state_dict(stage1_checkpoint["selector"])
            selector.eval()
            stage1_probs = torch.tensor(stage1_checkpoint["candidate_probs"], dtype=torch.float32)
            geometry_best = candidate_scores_raw.max()
            geometry_indices = torch.where(torch.abs(candidate_scores_raw - geometry_best) < 1e-6)[0].tolist()
            geometry_indices = sorted(geometry_indices, key=lambda i: float(stage1_probs[i]), reverse=True)

            sweep_rows = []
            best_candidate_idx = None
            best_candidate_state = None
            best_candidate_psnr = -float("inf")
            best_candidate_sam = float("inf")

            for cand_idx in geometry_indices:
                cand_a = int(candidate_ab[cand_idx, 0].item())
                cand_b = int(candidate_ab[cand_idx, 1].item())
                cand_mask = candidate_tiles[cand_idx].to(DEVICE)
                cand_model = UNet2D().to(DEVICE)
                cand_model.load_state_dict(stage1_checkpoint["model"])
                cand_optimizer = torch.optim.Adam(cand_model.parameters(), lr=REFINE_LR)
                cand_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(cand_optimizer, T_max=HARD_SWEEP_EPOCHS)
                cand_best_psnr = -float("inf")
                cand_best_sam = float("inf")
                cand_best_state = None

                for sweep_epoch in range(1, HARD_SWEEP_EPOCHS + 1):
                    cand_model.train()
                    for x in train_loader:
                        x = x.to(DEVICE)
                        sensed = apply_candidate_mask(x, cand_mask)
                        pred = cand_model(sensed)
                        loss = criterion(pred, x) + TRAIN_SAM_WEIGHT * spectral_angle_loss(pred, x)
                        cand_optimizer.zero_grad()
                        loss.backward()
                        cand_optimizer.step()

                    cand_scheduler.step()
                    cand_model.eval()
                    cand_val_psnr = 0.0
                    cand_val_sam = 0.0
                    with torch.no_grad():
                        for x in val_loader:
                            x = x.to(DEVICE)
                            pred = cand_model(apply_candidate_mask(x, cand_mask))
                            cand_val_psnr += compute_psnr(pred, x).item()
                            cand_val_sam += spectral_angle_mapper(pred, x).item()

                    cand_val_psnr /= len(val_loader)
                    cand_val_sam /= len(val_loader)
                    if (cand_val_psnr > cand_best_psnr) or (
                        abs(cand_val_psnr - cand_best_psnr) < 1e-6 and cand_val_sam < cand_best_sam
                    ):
                        cand_best_psnr = cand_val_psnr
                        cand_best_sam = cand_val_sam
                        cand_best_state = {k: v.detach().cpu().clone() for k, v in cand_model.state_dict().items()}

                sweep_rows.append(
                    {
                        "candidate_index": cand_idx,
                        "a": cand_a,
                        "b": cand_b,
                        "score_raw": float(candidate_scores_raw[cand_idx].item()),
                        "selector_probability": float(stage1_probs[cand_idx].item()),
                        "best_val_psnr": cand_best_psnr,
                        "best_val_sam_deg": cand_best_sam,
                    }
                )
                print(
                    f"Sweep ({cand_a},{cand_b}) | best val PSNR {cand_best_psnr:.2f} dB | "
                    f"best val SAM {cand_best_sam:.2f} deg"
                )

                if (cand_best_psnr > best_candidate_psnr) or (
                    abs(cand_best_psnr - best_candidate_psnr) < 1e-6 and cand_best_sam < best_candidate_sam
                ):
                    best_candidate_idx = cand_idx
                    best_candidate_state = cand_best_state
                    best_candidate_psnr = cand_best_psnr
                    best_candidate_sam = cand_best_sam

            SWEEP_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(SWEEP_HISTORY_PATH, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
                writer.writeheader()
                writer.writerows(sweep_rows)

            selected_idx = int(best_candidate_idx)
            selected_a = int(candidate_ab[selected_idx, 0].item())
            selected_b = int(candidate_ab[selected_idx, 1].item())
            selected_mask = candidate_tiles[selected_idx].to(DEVICE)

            model = UNet2D().to(DEVICE)
            model.load_state_dict(best_candidate_state)
            refine_optimizer = torch.optim.Adam(model.parameters(), lr=REFINE_LR)
            refine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(refine_optimizer, T_max=FINAL_REFINE_EPOCHS)
            refine_best_psnr = -float("inf")
            refine_history = []

            for epoch in range(1, FINAL_REFINE_EPOCHS + 1):
                model.train()
                refine_train_loss = 0.0
                refine_train_psnr = 0.0

                for x in train_loader:
                    x = x.to(DEVICE)
                    sensed = apply_candidate_mask(x, selected_mask)
                    pred = model(sensed)
                    loss = criterion(pred, x) + TRAIN_SAM_WEIGHT * spectral_angle_loss(pred, x)
                    refine_optimizer.zero_grad()
                    loss.backward()
                    refine_optimizer.step()
                    refine_train_loss += loss.item()
                    refine_train_psnr += compute_psnr(pred, x).item()

                refine_scheduler.step()
                model.eval()
                refine_val_psnr = 0.0
                refine_val_sam = 0.0
                with torch.no_grad():
                    for x in val_loader:
                        x = x.to(DEVICE)
                        pred = model(apply_candidate_mask(x, selected_mask))
                        refine_val_psnr += compute_psnr(pred, x).item()
                        refine_val_sam += spectral_angle_mapper(pred, x).item()

                refine_train_loss /= len(train_loader)
                refine_train_psnr /= len(train_loader)
                refine_val_psnr /= len(val_loader)
                refine_val_sam /= len(val_loader)
                refine_history.append(
                    {
                        "epoch": epoch,
                        "train_loss": refine_train_loss,
                        "train_psnr": refine_train_psnr,
                        "val_psnr": refine_val_psnr,
                        "val_sam_deg": refine_val_sam,
                        "selected_index": selected_idx,
                        "selected_a": selected_a,
                        "selected_b": selected_b,
                        "selected_score_raw": float(candidate_scores_raw[selected_idx].item()),
                    }
                )
                print(
                    f"Refine {epoch:02d} | train PSNR {refine_train_psnr:.2f} dB | "
                    f"val PSNR {refine_val_psnr:.2f} dB | val SAM {refine_val_sam:.2f} deg | "
                    f"fixed sel ({selected_a},{selected_b}) | lr {refine_scheduler.get_last_lr()[0]:.2e}"
                )

                if refine_val_psnr > refine_best_psnr:
                    refine_best_psnr = refine_val_psnr
                    REFINE_CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model": model.state_dict(),
                            "selector": selector.state_dict(),
                            "best_val_psnr": refine_best_psnr,
                            "selected_index": selected_idx,
                            "selected_a": selected_a,
                            "selected_b": selected_b,
                            "selected_score_raw": float(candidate_scores_raw[selected_idx].item()),
                            "geometry_optimal_indices": geometry_indices,
                            "sweep_rows": sweep_rows,
                            "candidate_ab": candidate_ab.cpu().numpy(),
                            "candidate_scores_raw": candidate_scores_raw.cpu().numpy(),
                            "candidate_probs": stage1_checkpoint["candidate_probs"],
                            "hard_tile": (candidate_tiles[selected_idx].argmax(dim=0) + 1).detach().cpu().numpy(),
                        },
                        REFINE_CKPT_PATH,
                    )

            REFINE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(REFINE_HISTORY_PATH, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(refine_history[0].keys()))
                writer.writeheader()
                writer.writerows(refine_history)
            """
        ),
        code_cell(
            """
            checkpoint = torch.load(REFINE_CKPT_PATH, map_location=DEVICE, weights_only=False)
            selected_idx = int(checkpoint["selected_index"])
            selected_a = int(checkpoint["selected_a"])
            selected_b = int(checkpoint["selected_b"])
            hard_tile = checkpoint["hard_tile"]
            probs = checkpoint["candidate_probs"]
            candidate_ab_np = checkpoint["candidate_ab"]
            candidate_scores_np = checkpoint["candidate_scores_raw"]
            sweep_rows = checkpoint.get("sweep_rows", [])

            best_osp_idx = int(np.argmax(candidate_scores_np))
            best_osp_ab = tuple(int(v) for v in candidate_ab_np[best_osp_idx].tolist())

            labels = [f"({int(a)},{int(b)})" for a, b in candidate_ab_np.tolist()]
            top_order = np.argsort(probs)[::-1]

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(hard_tile, cmap="tab20")
            plt.title(f"Learned exact OSP tile\\nselected (a,b)=({selected_a},{selected_b})")
            plt.colorbar()

            plt.subplot(1, 2, 2)
            plt.bar(range(len(probs)), probs, color="tab:blue")
            plt.xticks(range(len(probs)), labels, rotation=45, ha="right")
            plt.title("Candidate selection probabilities")
            plt.ylabel("Probability")
            plt.tight_layout()
            FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
            plt.show()

            print("Exact best OSP candidate:", best_osp_ab, "score", float(candidate_scores_np[best_osp_idx]))
            print("Learned selected candidate:", (selected_a, selected_b), "score", float(checkpoint["selected_score_raw"]))
            print("Top-3 learned candidates:", [(labels[i], float(probs[i])) for i in top_order[:3]])
            print("Best refined val PSNR:", float(checkpoint["best_val_psnr"]))
            if sweep_rows:
                print("Hard-sweep summary:", sweep_rows)
            """
        ),
        md_cell(
            """
            Use this notebook when the final filter must remain inside the exact discrete OSP family.

            It is the strictest answer to:
            "Can the OSP parameter-selection step itself be made learnable?"
            """
        ),
    ]
)

notebooks["11_phase11_osp_seeded_learnable_msfa.ipynb"] = notebook(
    [
        md_cell(
            """
            # Phase 11: Fair OSP vs Learnable Pipeline

            Goal: run fixed OSP and OSP-seeded learnable MSFA in one notebook with fair settings:
            - same dataset
            - same UNet
            - same epoch budget and loop structure
            - explicit evaluation, visualization, and result table
            """
        ),
        md_cell(common_header),
        colab_mount_cell(),
        code_cell(
            """
            import csv
            import math
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            from torch.utils.data import DataLoader, Dataset

            PROJECT_ROOT = Path("/content/drive/MyDrive/Msa-Osp")
            PATCH_PATH = PROJECT_ROOT / "dataset_patches.npz"
            OSP_REFINE_CKPT_PATH = PROJECT_ROOT / "learned_osp_selector_refine_best.pth"
            OSP_SELECTOR_CKPT_PATH = PROJECT_ROOT / "learned_osp_selector_best.pth"
            LEARNED3D_CKPT_PATH = PROJECT_ROOT / "learned_msfa_3dsp_best.pth"
            BASELINE_CKPT_PATH = PROJECT_ROOT / "unet_baseline_best.pth"
            # Notebook run mode.
            MODE = "both"  # options: "osp", "learnable", "both"

            HISTORY_OSP_PATH = PROJECT_ROOT / "phase11_osp_history.csv"
            HISTORY_EXACT_PATH = PROJECT_ROOT / "phase11_exact_selector_history.csv"
            HISTORY_LEARN_PATH = PROJECT_ROOT / "phase11_learnable_history.csv"
            CKPT_OSP_PATH = PROJECT_ROOT / "phase11_osp_best.pth"
            CKPT_EXACT_PATH = PROJECT_ROOT / "phase11_exact_selector_best.pth"
            CKPT_LEARN_PATH = PROJECT_ROOT / "phase11_learnable_best.pth"
            TILE_LEARN_SOFT_PATH = PROJECT_ROOT / "phase11_learnable_tile_soft.npy"
            SNAPSHOT_DIR = PROJECT_ROOT / "phase11_tile_snapshots"
            FIG_DIR = PROJECT_ROOT / "phase11_figures"

            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            BATCH_SIZE = 8
            EPOCHS = 72
            SELECTOR_EPOCHS = 30
            EXACT_REFINE_EPOCHS = 12
            UNET_LR = 1.5e-4
            MSFA_LR = 5e-4
            SELECTOR_LR = 1.5e-2
            EXACT_REFINE_LR = 5e-5
            BAND_COUNT = 16
            TILE_SIZE = 16
            BASE = 32
            TEMP_START = 1.00
            TEMP_END = 0.20
            SELECTOR_TEMP_START = 2.0
            SELECTOR_TEMP_END = 0.25
            SELECTOR_REHEAT_START = 14
            SELECTOR_REHEAT_LENGTH = 4
            SELECTOR_REHEAT_TEMP = 1.0
            INIT_LOGIT_BOOST = 2.0
            DELTA_INIT_STD = 5e-2
            PRIOR_SCALE_START = 1.00
            PRIOR_SCALE_END = 0.30
            PRIOR_RELEASE_EPOCHS = 28
            SAM_WEIGHT = 5e-2
            LAMBDA_SP = 1e-3
            LAMBDA_ROWCOL = 2e-3
            LAMBDA_DELTA = 5e-6
            LAMBDA_ENT_EXPLORE = -1.0e-3
            LAMBDA_ENT_FINAL = 7e-4
            LAMBDA_ESCAPE = 2e-2
            ESCAPE_EPOCHS = 18
            TARGET_MATCH_START = 0.98
            TARGET_MATCH_END = 0.88
            D_MIN_3D = 0.55
            Z_WEIGHT = 1.25
            GRAD_WEIGHT = 1e-2
            SPEC_WEIGHT = 2e-2
            LAMBDA_SCORE = 5e-2
            LAMBDA_SELECTOR_ENT = 2e-3
            DISTILL_WEIGHT_START = 5e-2
            DISTILL_WEIGHT_END = 0.0
            CKPT_SCORE_SAM_WEIGHT = 0.15
            MIN_LR = 1e-5
            EARLY_STOP_PATIENCE = 9

            try:
                from skimage.metrics import structural_similarity as ssim_fn
                HAS_SSIM = True
            except Exception:
                HAS_SSIM = False

            print("Device:", DEVICE)
            print("MODE:", MODE)
            """
        ),
        code_cell(
            """
            data = np.load(PATCH_PATH)
            train_target = data["train"]
            val_target = data["val"]

            class CubeDataset(Dataset):
                def __init__(self, cubes):
                    self.cubes = cubes

                def __len__(self):
                    return len(self.cubes)

                def __getitem__(self, idx):
                    return torch.from_numpy(self.cubes[idx]).permute(2, 0, 1).float()

            train_loader = DataLoader(CubeDataset(train_target), batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(CubeDataset(val_target), batch_size=BATCH_SIZE, shuffle=False)
            """
        ),
        code_cell(
            """
            class DoubleConv(nn.Module):
                def __init__(self, in_ch, out_ch):
                    super().__init__()
                    self.block = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_ch, out_ch, 3, padding=1),
                        nn.ReLU(inplace=True),
                    )

                def forward(self, x):
                    return self.block(x)

            class UNet2D(nn.Module):
                def __init__(self, in_ch=1, out_ch=16, base=BASE):
                    super().__init__()
                    self.enc1 = DoubleConv(in_ch, base)
                    self.pool1 = nn.MaxPool2d(2)
                    self.enc2 = DoubleConv(base, base * 2)
                    self.pool2 = nn.MaxPool2d(2)
                    self.bottleneck = DoubleConv(base * 2, base * 4)
                    self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
                    self.dec2 = DoubleConv(base * 4, base * 2)
                    self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
                    self.dec1 = DoubleConv(base * 2, base)
                    self.final = nn.Conv2d(base, out_ch, 1)

                def forward(self, x):
                    e1 = self.enc1(x)
                    e2 = self.enc2(self.pool1(e1))
                    b = self.bottleneck(self.pool2(e2))
                    d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
                    d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
                    return self.final(d1)

            def compute_psnr(pred, target, eps=1e-8):
                mse = torch.mean((pred - target) ** 2)
                return 20 * torch.log10(1.0 / torch.sqrt(mse + eps))

            def spectral_angle_mapper(pred, target, eps=1e-8):
                dot = torch.sum(pred * target, dim=1)
                pred_norm = torch.norm(pred, dim=1)
                target_norm = torch.norm(target, dim=1)
                cos_theta = torch.clamp(dot / (pred_norm * target_norm + eps), -1 + eps, 1 - eps)
                return torch.rad2deg(torch.acos(cos_theta)).mean()

            def spectral_angle_loss(pred, target, eps=1e-8):
                dot = torch.sum(pred * target, dim=1)
                pred_norm = torch.norm(pred, dim=1)
                target_norm = torch.norm(target, dim=1)
                cos_theta = torch.clamp(dot / (pred_norm * target_norm + eps), -1 + eps, 1 - eps)
                return torch.acos(cos_theta).mean()

            def compute_rgb_ssim(pred, target):
                if not HAS_SSIM:
                    return float("nan")
                pred_np = pred.detach().cpu().numpy()
                target_np = target.detach().cpu().numpy()
                scores = []
                for i in range(pred_np.shape[0]):
                    p = np.transpose(pred_np[i, [5, 10, 15]], (1, 2, 0))
                    t = np.transpose(target_np[i, [5, 10, 15]], (1, 2, 0))
                    data_range = max(float(t.max() - t.min()), 1e-8)
                    scores.append(ssim_fn(t, p, data_range=data_range, channel_axis=2))
                return float(np.mean(scores))

            def spatial_gradient_loss(pred, target):
                pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
                pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
                target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
                target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
                return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

            def spectral_gradient_consistency_loss(pred, target):
                pred_dlambda = pred[:, 1:, :, :] - pred[:, :-1, :, :]
                target_dlambda = target[:, 1:, :, :] - target[:, :-1, :, :]
                return F.l1_loss(pred_dlambda, target_dlambda)

            def build_candidate_bank():
                idx = torch.arange(1, BAND_COUNT + 1, dtype=torch.int64)
                I, J = torch.meshgrid(idx, idx, indexing="ij")
                rows = []
                for a in range(1, BAND_COUNT // 2 + 1):
                    for b in range(a, BAND_COUNT // 2 + 1):
                        if math.gcd(a, BAND_COUNT) != 1 or math.gcd(b, BAND_COUNT) != 1:
                            continue
                        tile = torch.remainder(I * a + J * b, BAND_COUNT) + 1
                        if len(torch.unique(tile)) != BAND_COUNT:
                            continue
                        target = set(range(1, BAND_COUNT + 1))
                        if not all(set(tile[r, :].tolist()) == target for r in range(TILE_SIZE)):
                            continue
                        if not all(set(tile[:, c].tolist()) == target for c in range(TILE_SIZE)):
                            continue
                        rows.append((a, b, tile))
                return rows

            def choose_osp_reference():
                for path in [OSP_REFINE_CKPT_PATH, OSP_SELECTOR_CKPT_PATH]:
                    if path.exists():
                        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                        if "hard_tile" in checkpoint:
                            tile = torch.tensor(checkpoint["hard_tile"], dtype=torch.int64)
                            return tile, (int(checkpoint["selected_a"]), int(checkpoint["selected_b"])), path.name

                candidates = build_candidate_bank()
                fallback = next((tile for a, b, tile in candidates if (a, b) == (3, 5)), None)
                if fallback is None:
                    fallback = candidates[0][2]
                    ab = (int(candidates[0][0]), int(candidates[0][1]))
                else:
                    ab = (3, 5)
                return fallback, ab, "geometry_fallback"

            candidate_records = []
            for a, b, tile in build_candidate_bank():
                candidate_records.append(
                    {
                        "a": int(a),
                        "b": int(b),
                        "tile": tile,
                        "score_raw": torch.pdist(
                            torch.stack(
                                [
                                    torch.meshgrid(
                                        torch.arange(1, TILE_SIZE + 1, dtype=torch.float32),
                                        torch.arange(1, TILE_SIZE + 1, dtype=torch.float32),
                                        indexing="ij",
                                    )[0].reshape(-1),
                                    torch.meshgrid(
                                        torch.arange(1, TILE_SIZE + 1, dtype=torch.float32),
                                        torch.arange(1, TILE_SIZE + 1, dtype=torch.float32),
                                        indexing="ij",
                                    )[1].reshape(-1),
                                    tile.float().reshape(-1),
                                ],
                                dim=1,
                            )
                        ).min().item(),
                    }
                )

            candidate_tiles = torch.stack(
                [F.one_hot((rec["tile"] - 1).long(), num_classes=BAND_COUNT).permute(2, 0, 1).float() for rec in candidate_records],
                dim=0,
            )
            candidate_scores_raw = torch.tensor([rec["score_raw"] for rec in candidate_records], dtype=torch.float32)
            candidate_scores_norm = (candidate_scores_raw - candidate_scores_raw.min()) / (
                candidate_scores_raw.max() - candidate_scores_raw.min() + 1e-8
            )
            candidate_ab = torch.tensor([[rec["a"], rec["b"]] for rec in candidate_records], dtype=torch.int64)

            init_tile_one_based, init_ab, init_source = choose_osp_reference()
            print("OSP reference source:", init_source)
            print("OSP reference (a,b):", init_ab)

            class FixedOSP(nn.Module):
                def __init__(self, init_tile_one_based):
                    super().__init__()
                    init_tile_zero_based = init_tile_one_based.long() - 1
                    onehot = F.one_hot(init_tile_zero_based, num_classes=BAND_COUNT).permute(2, 0, 1).float()
                    self.register_buffer("weights", onehot)
                    self.register_buffer("init_tile_one_based", init_tile_one_based.long())

                def soft_tile(self, temp=None, prior_scale=None):
                    return self.weights

                def hard_tile(self, temp=None, prior_scale=None):
                    return self.init_tile_one_based

                def forward(self, x, temp=None, prior_scale=None):
                    _, _, h, w = x.shape
                    weights_full = self.weights.repeat(1, h // TILE_SIZE, w // TILE_SIZE)
                    return (x * weights_full.unsqueeze(0)).sum(dim=1, keepdim=True)

            class OSPSeededLearnableMSFA(nn.Module):
                def __init__(self, init_tile_one_based):
                    super().__init__()
                    init_tile_zero_based = init_tile_one_based.long() - 1
                    prior = F.one_hot(init_tile_zero_based, num_classes=BAND_COUNT).permute(2, 0, 1).float()
                    prior_logits = torch.full_like(prior, -INIT_LOGIT_BOOST)
                    prior_logits = prior_logits + prior * (2.0 * INIT_LOGIT_BOOST)
                    self.register_buffer("prior_logits", prior_logits)
                    self.register_buffer("init_tile_one_based", init_tile_one_based.long())
                    self.register_buffer("init_onehot", prior)
                    self.delta_logits = nn.Parameter(DELTA_INIT_STD * torch.randn_like(prior_logits))

                def logits(self, prior_scale=1.0):
                    return prior_scale * self.prior_logits + self.delta_logits

                def soft_tile(self, temp, prior_scale=1.0):
                    return torch.softmax(self.logits(prior_scale=prior_scale) / temp, dim=0)

                def hard_tile(self, temp=TEMP_END, prior_scale=PRIOR_SCALE_END):
                    return self.soft_tile(temp=temp, prior_scale=prior_scale).argmax(dim=0) + 1

                def forward(self, x, temp, prior_scale=1.0):
                    _, _, h, w = x.shape
                    weights = self.soft_tile(temp=temp, prior_scale=prior_scale)
                    weights_full = weights.repeat(1, h // TILE_SIZE, w // TILE_SIZE)
                    return (x * weights_full.unsqueeze(0)).sum(dim=1, keepdim=True)

            class ExactOSPSelector(nn.Module):
                def __init__(self, candidate_tiles, candidate_scores_raw, candidate_scores_norm, candidate_ab):
                    super().__init__()
                    self.register_buffer("candidate_tiles", candidate_tiles)
                    self.register_buffer("candidate_scores_raw", candidate_scores_raw)
                    self.register_buffer("candidate_scores_norm", candidate_scores_norm)
                    self.register_buffer("candidate_ab", candidate_ab)
                    self.logits = nn.Parameter(0.25 * candidate_scores_norm.clone())

                def probs(self, temp, training=True):
                    if training:
                        return F.gumbel_softmax(self.logits, tau=temp, hard=False, dim=0)
                    return torch.softmax(self.logits / temp, dim=0)

                def soft_tile(self, temp, training=True):
                    probs = self.probs(temp=temp, training=training)
                    return torch.sum(probs.view(-1, 1, 1, 1) * self.candidate_tiles, dim=0)

                def hard_index(self):
                    return int(torch.argmax(self.logits).item())

                def hard_tile(self):
                    return self.candidate_tiles[self.hard_index()].argmax(dim=0) + 1

                def selected_ab(self):
                    idx = self.hard_index()
                    a, b = self.candidate_ab[idx].tolist()
                    return idx, (int(a), int(b))

                def selected_score(self):
                    return float(self.candidate_scores_raw[self.hard_index()].item())

                def selector_entropy(self, temp, training=True, eps=1e-8):
                    probs = self.probs(temp=temp, training=training)
                    return -(probs * torch.log(probs + eps)).sum()

                def expected_score(self, temp, training=True):
                    probs = self.probs(temp=temp, training=training)
                    return torch.sum(probs * self.candidate_scores_norm)

                def forward(self, x, temp, training=True):
                    _, _, h, w = x.shape
                    weights = self.soft_tile(temp=temp, training=training)
                    weights_full = weights.repeat(1, h // TILE_SIZE, w // TILE_SIZE)
                    return (x * weights_full.unsqueeze(0)).sum(dim=1, keepdim=True)

            spatial_coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(0.0, 1.0, TILE_SIZE),
                    torch.linspace(0.0, 1.0, TILE_SIZE),
                    indexing="ij",
                ),
                dim=-1,
            ).reshape(-1, 2)
            wavelength_coords = torch.linspace(0.0, 1.0, BAND_COUNT)

            def osp_3d_points(msfa, temp, prior_scale):
                soft_tile = msfa.soft_tile(temp=temp, prior_scale=prior_scale)
                z = (soft_tile * wavelength_coords.to(soft_tile.device).view(BAND_COUNT, 1, 1)).sum(dim=0).reshape(-1, 1)
                xy = spatial_coords.to(soft_tile.device)
                return torch.cat([xy, Z_WEIGHT * z], dim=1)

            def osp_3d_loss(msfa, temp, prior_scale):
                points = osp_3d_points(msfa, temp, prior_scale)
                d = torch.cdist(points, points, p=2)
                mask = torch.triu(torch.ones_like(d), diagonal=1) > 0
                distances = d[mask]
                return torch.relu(D_MIN_3D - distances).mean()

            def distance_stats(msfa, temp, prior_scale):
                points = osp_3d_points(msfa, temp, prior_scale)
                d = torch.cdist(points, points, p=2)
                mask = torch.triu(torch.ones_like(d), diagonal=1) > 0
                distances = d[mask]
                return {
                    "min_3d_distance": distances.min().item(),
                    "mean_3d_distance": distances.mean().item(),
                }

            def rowcol_uniformity_loss(msfa, temp, prior_scale):
                soft_tile = msfa.soft_tile(temp=temp, prior_scale=prior_scale)
                row_counts = soft_tile.sum(dim=2)
                col_counts = soft_tile.sum(dim=1)
                target_row = torch.ones_like(row_counts)
                target_col = torch.ones_like(col_counts)
                return ((row_counts - target_row) ** 2).mean() + ((col_counts - target_col) ** 2).mean()

            def delta_regularization(msfa):
                return msfa.delta_logits.pow(2).mean()

            def entropy_loss(msfa, temp, prior_scale, eps=1e-8):
                soft_tile = msfa.soft_tile(temp=temp, prior_scale=prior_scale)
                return -(soft_tile * torch.log(soft_tile + eps)).sum(dim=0).mean()

            def seed_match(msfa, temp, prior_scale):
                soft_tile = msfa.soft_tile(temp=temp, prior_scale=prior_scale)
                return (soft_tile * msfa.init_onehot).sum(dim=0).mean()

            def escape_from_seed_loss(msfa, temp, prior_scale, epoch):
                if epoch > ESCAPE_EPOCHS:
                    return torch.zeros((), device=msfa.delta_logits.device)
                progress = (epoch - 1) / max(ESCAPE_EPOCHS - 1, 1)
                target_match = TARGET_MATCH_START + progress * (TARGET_MATCH_END - TARGET_MATCH_START)
                return torch.relu(seed_match(msfa, temp=temp, prior_scale=prior_scale) - target_match)

            def changed_fraction(msfa, temp, prior_scale):
                hard_tile = msfa.hard_tile(temp=temp, prior_scale=prior_scale)
                return (hard_tile != msfa.init_tile_one_based).float().mean().item()

            def try_load_unet_weights(model):
                for path in [BASELINE_CKPT_PATH, OSP_REFINE_CKPT_PATH, OSP_SELECTOR_CKPT_PATH, LEARNED3D_CKPT_PATH]:
                    if not path.exists():
                        continue
                    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                    state = None
                    if "model" in checkpoint:
                        state = checkpoint["model"]
                    elif "unet" in checkpoint:
                        state = checkpoint["unet"]
                    elif "model_state_dict" in checkpoint:
                        state = checkpoint["model_state_dict"]
                    if state is None:
                        continue
                    try:
                        model.load_state_dict(state, strict=True)
                        print("Loaded UNet weights from:", path.name)
                        return path.name
                    except Exception:
                        continue
                print("UNet warm-start skipped.")
                return None

            def evaluate(msfa, model, dataloader, mode_name, temp=TEMP_END, prior_scale=PRIOR_SCALE_END):
                model.eval()
                msfa.eval()
                psnr_total = 0.0
                sam_total = 0.0
                ssim_total = 0.0
                with torch.no_grad():
                    for x in dataloader:
                        x = x.to(DEVICE)
                        if mode_name == "learnable":
                            sensed = msfa(x, temp=temp, prior_scale=prior_scale)
                        else:
                            sensed = msfa(x)
                        pred = model(sensed)
                        psnr_total += compute_psnr(pred, x).item()
                        sam_total += spectral_angle_mapper(pred, x).item()
                        ssim_total += compute_rgb_ssim(pred, x)
                n = len(dataloader)
                return psnr_total / n, sam_total / n, ssim_total / n
            """
        ),
        code_cell(
            """
            def temperature(epoch):
                return TEMP_START + (TEMP_END - TEMP_START) * (epoch - 1) / max(EPOCHS - 1, 1)

            def prior_scale(epoch):
                if epoch >= PRIOR_RELEASE_EPOCHS:
                    return PRIOR_SCALE_END
                return PRIOR_SCALE_START + (PRIOR_SCALE_END - PRIOR_SCALE_START) * (epoch - 1) / max(PRIOR_RELEASE_EPOCHS - 1, 1)

            def rowcol_weight(epoch):
                progress = (epoch - 1) / max(EPOCHS - 1, 1)
                return LAMBDA_ROWCOL * max(0.0, min(1.0, (progress - 0.20) / 0.35))

            def delta_weight(epoch):
                progress = (epoch - 1) / max(EPOCHS - 1, 1)
                return LAMBDA_DELTA * max(0.0, min(1.0, (progress - 0.55) / 0.25))

            def entropy_weight(epoch):
                progress = (epoch - 1) / max(EPOCHS - 1, 1)
                if progress <= 0.40:
                    return LAMBDA_ENT_EXPLORE
                ramp = min(1.0, (progress - 0.40) / 0.60)
                return LAMBDA_ENT_EXPLORE + ramp * (LAMBDA_ENT_FINAL - LAMBDA_ENT_EXPLORE)

            def selector_temperature(epoch):
                base = SELECTOR_TEMP_START + (SELECTOR_TEMP_END - SELECTOR_TEMP_START) * (epoch - 1) / max(SELECTOR_EPOCHS - 1, 1)
                if SELECTOR_REHEAT_START <= epoch < SELECTOR_REHEAT_START + SELECTOR_REHEAT_LENGTH:
                    frac = (epoch - SELECTOR_REHEAT_START) / max(SELECTOR_REHEAT_LENGTH - 1, 1)
                    return SELECTOR_REHEAT_TEMP + frac * (base - SELECTOR_REHEAT_TEMP)
                return base

            def distill_weight(epoch):
                progress = (epoch - 1) / max(EPOCHS - 1, 1)
                return DISTILL_WEIGHT_START + progress * (DISTILL_WEIGHT_END - DISTILL_WEIGHT_START)

            def save_epoch_tile(msfa, mode_name, epoch, temp, pscale):
                SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
                mode_dir = SNAPSHOT_DIR / mode_name
                mode_dir.mkdir(parents=True, exist_ok=True)
                if mode_name == "learnable":
                    tile = msfa.hard_tile(temp=temp, prior_scale=pscale).detach().cpu().numpy()
                elif mode_name == "exact_selector":
                    tile = msfa.hard_tile().detach().cpu().numpy()
                else:
                    tile = msfa.hard_tile().detach().cpu().numpy()
                np.save(mode_dir / f"hard_tile_epoch_{epoch:03d}.npy", tile)

            def load_model_state_if_present(model, state_dict):
                if state_dict is not None:
                    model.load_state_dict(state_dict, strict=True)
                    return "inline_warm_start"
                return try_load_unet_weights(model)

            def train_exact_selector(model_state=None):
                model = UNet2D().to(DEVICE)
                warm_start_source = load_model_state_if_present(model, model_state)
                selector = ExactOSPSelector(
                    candidate_tiles=candidate_tiles.to(DEVICE),
                    candidate_scores_raw=candidate_scores_raw.to(DEVICE),
                    candidate_scores_norm=candidate_scores_norm.to(DEVICE),
                    candidate_ab=candidate_ab.to(DEVICE),
                ).to(DEVICE)
                optimizer = torch.optim.Adam(
                    [{"params": model.parameters(), "lr": UNET_LR}, {"params": selector.parameters(), "lr": SELECTOR_LR}]
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SELECTOR_EPOCHS, eta_min=MIN_LR)
                criterion = nn.L1Loss()
                best_psnr = -float("inf")
                best_quality = -float("inf")
                epochs_since_improve = 0
                history = []

                for epoch in range(1, SELECTOR_EPOCHS + 1):
                    temp = selector_temperature(epoch)
                    model.train()
                    selector.train()
                    train_loss = 0.0
                    train_psnr = 0.0
                    for x in train_loader:
                        x = x.to(DEVICE)
                        pred = model(selector(x, temp=temp, training=True))
                        recon_l1 = criterion(pred, x)
                        recon_sam = spectral_angle_loss(pred, x)
                        recon_grad = spatial_gradient_loss(pred, x)
                        recon_spec = spectral_gradient_consistency_loss(pred, x)
                        score_loss = -selector.expected_score(temp=temp, training=True)
                        ent_loss = selector.selector_entropy(temp=temp, training=True)
                        loss = (
                            recon_l1
                            + SAM_WEIGHT * recon_sam
                            + GRAD_WEIGHT * recon_grad
                            + SPEC_WEIGHT * recon_spec
                            + LAMBDA_SCORE * score_loss
                            + LAMBDA_SELECTOR_ENT * ent_loss
                        )
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                        train_psnr += compute_psnr(pred, x).item()

                    model.eval()
                    selector.eval()
                    val_psnr = 0.0
                    val_sam = 0.0
                    with torch.no_grad():
                        for x in val_loader:
                            x = x.to(DEVICE)
                            pred = model(selector(x, temp=temp, training=False))
                            val_psnr += compute_psnr(pred, x).item()
                            val_sam += spectral_angle_mapper(pred, x).item()

                    train_loss /= len(train_loader)
                    train_psnr /= len(train_loader)
                    val_psnr /= len(val_loader)
                    val_sam /= len(val_loader)
                    quality = val_psnr - CKPT_SCORE_SAM_WEIGHT * val_sam
                    selected_idx, (selected_a, selected_b) = selector.selected_ab()
                    top_prob = selector.probs(temp=temp, training=False).max().item()
                    history.append(
                        {
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "train_psnr": train_psnr,
                            "val_psnr": val_psnr,
                            "val_sam_deg": val_sam,
                            "temperature": temp,
                            "selected_index": selected_idx,
                            "selected_a": selected_a,
                            "selected_b": selected_b,
                            "selected_score_raw": selector.selected_score(),
                            "top_probability": top_prob,
                        }
                    )
                    print(
                        f"[exact] Epoch {epoch:02d} | train PSNR {train_psnr:.2f} dB | "
                        f"val PSNR {val_psnr:.2f} dB | val SAM {val_sam:.2f} deg | "
                        f"sel ({selected_a},{selected_b}) | top p {top_prob:.3f}"
                    )
                    scheduler.step()

                    if epoch % 5 == 0:
                        save_epoch_tile(selector, "exact_selector", epoch, temp=temp, pscale=1.0)

                    if quality > best_quality:
                        best_psnr = val_psnr
                        best_quality = quality
                        epochs_since_improve = 0
                        torch.save(
                            {
                                "epoch": epoch,
                                "model": model.state_dict(),
                                "selector": selector.state_dict(),
                                "best_val_psnr": best_psnr,
                                "best_quality": best_quality,
                                "temperature": temp,
                                "selected_index": selected_idx,
                                "selected_a": selected_a,
                                "selected_b": selected_b,
                                "selected_score_raw": selector.selected_score(),
                                "hard_tile": selector.hard_tile().detach().cpu().numpy(),
                                "warm_start_source": warm_start_source,
                            },
                            CKPT_EXACT_PATH,
                        )
                    else:
                        epochs_since_improve += 1
                        if epochs_since_improve >= EARLY_STOP_PATIENCE:
                            print(f"[exact] Early stopping at epoch {epoch:02d}")
                            break

                HISTORY_EXACT_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(HISTORY_EXACT_PATH, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
                    writer.writeheader()
                    writer.writerows(history)

                exact_ckpt = torch.load(CKPT_EXACT_PATH, map_location=DEVICE, weights_only=False)
                selected_tile = torch.tensor(exact_ckpt["hard_tile"], dtype=torch.int64)
                selected_msfa = FixedOSP(init_tile_one_based=selected_tile).to(DEVICE)
                model.load_state_dict(exact_ckpt["model"], strict=True)

                refine_optimizer = torch.optim.Adam(model.parameters(), lr=EXACT_REFINE_LR)
                refine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(refine_optimizer, T_max=EXACT_REFINE_EPOCHS, eta_min=MIN_LR)
                refine_best_psnr = -float("inf")
                refine_best_quality = -float("inf")
                for epoch in range(1, EXACT_REFINE_EPOCHS + 1):
                    model.train()
                    for x in train_loader:
                        x = x.to(DEVICE)
                        pred = model(selected_msfa(x))
                        loss = (
                            criterion(pred, x)
                            + SAM_WEIGHT * spectral_angle_loss(pred, x)
                            + GRAD_WEIGHT * spatial_gradient_loss(pred, x)
                            + SPEC_WEIGHT * spectral_gradient_consistency_loss(pred, x)
                        )
                        refine_optimizer.zero_grad()
                        loss.backward()
                        refine_optimizer.step()
                    refine_scheduler.step()
                    model.eval()
                    val_psnr = 0.0
                    val_sam = 0.0
                    with torch.no_grad():
                        for x in val_loader:
                            x = x.to(DEVICE)
                            pred = model(selected_msfa(x))
                            val_psnr += compute_psnr(pred, x).item()
                            val_sam += spectral_angle_mapper(pred, x).item()
                    val_psnr /= len(val_loader)
                    val_sam /= len(val_loader)
                    quality = val_psnr - CKPT_SCORE_SAM_WEIGHT * val_sam
                    print(
                        f"[exact-refine] Epoch {epoch:02d} | val PSNR {val_psnr:.2f} dB | "
                        f"val SAM {val_sam:.2f} deg"
                    )
                    if quality > refine_best_quality:
                        refine_best_psnr = val_psnr
                        refine_best_quality = quality
                        exact_ckpt["model"] = model.state_dict()
                        exact_ckpt["best_val_psnr"] = refine_best_psnr
                        exact_ckpt["best_quality"] = refine_best_quality
                        exact_ckpt["refined_val_sam_deg"] = val_sam
                        torch.save(exact_ckpt, CKPT_EXACT_PATH)

                exact_ckpt = torch.load(CKPT_EXACT_PATH, map_location=DEVICE, weights_only=False)
                return {
                    "selected_tile": torch.tensor(exact_ckpt["hard_tile"], dtype=torch.int64),
                    "selected_ab": (int(exact_ckpt["selected_a"]), int(exact_ckpt["selected_b"])),
                    "model_state": exact_ckpt["model"],
                    "ckpt_path": str(CKPT_EXACT_PATH),
                }

            def train_mode(mode_name, init_tile_for_mode=None, warm_model_state=None, teacher_bundle=None):
                assert mode_name in ["osp", "learnable"]
                history_path = HISTORY_OSP_PATH if mode_name == "osp" else HISTORY_LEARN_PATH
                ckpt_path = CKPT_OSP_PATH if mode_name == "osp" else CKPT_LEARN_PATH

                model = UNet2D().to(DEVICE)
                warm_start_source = load_model_state_if_present(model, warm_model_state)
                init_tile_local = init_tile_one_based if init_tile_for_mode is None else init_tile_for_mode
                if mode_name == "osp":
                    msfa = FixedOSP(init_tile_one_based=init_tile_local).to(DEVICE)
                    optimizer = torch.optim.Adam(model.parameters(), lr=UNET_LR)
                else:
                    msfa = OSPSeededLearnableMSFA(init_tile_one_based=init_tile_local).to(DEVICE)
                    optimizer = torch.optim.Adam(
                        [{"params": model.parameters(), "lr": UNET_LR}, {"params": msfa.delta_logits, "lr": MSFA_LR}]
                    )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)

                criterion = nn.L1Loss()
                best_psnr = -float("inf")
                best_quality = -float("inf")
                epochs_since_improve = 0
                history = []
                train_loss_list = []
                val_psnr_list = []

                for epoch in range(1, EPOCHS + 1):
                    temp = temperature(epoch)
                    pscale = prior_scale(epoch)
                    rc_w = rowcol_weight(epoch)
                    d_w = delta_weight(epoch)
                    ent_w = entropy_weight(epoch)
                    model.train()
                    msfa.train()
                    train_loss = 0.0
                    train_psnr = 0.0

                    for x in train_loader:
                        x = x.to(DEVICE)
                        sensed = msfa(x) if mode_name == "osp" else msfa(x, temp=temp, prior_scale=pscale)
                        pred = model(sensed)
                        recon_l1 = criterion(pred, x)
                        recon_sam = spectral_angle_loss(pred, x)
                        recon_grad = spatial_gradient_loss(pred, x)
                        recon_spec = spectral_gradient_consistency_loss(pred, x)
                        loss = recon_l1 + SAM_WEIGHT * recon_sam + GRAD_WEIGHT * recon_grad + SPEC_WEIGHT * recon_spec

                        sp_loss = torch.zeros((), device=DEVICE)
                        rc_loss = torch.zeros((), device=DEVICE)
                        delta_loss = torch.zeros((), device=DEVICE)
                        ent_loss = torch.zeros((), device=DEVICE)
                        esc_loss = torch.zeros((), device=DEVICE)
                        distill_loss = torch.zeros((), device=DEVICE)

                        if mode_name == "learnable":
                            sp_loss = osp_3d_loss(msfa, temp=temp, prior_scale=pscale)
                            rc_loss = rowcol_uniformity_loss(msfa, temp=temp, prior_scale=pscale)
                            delta_loss = delta_regularization(msfa)
                            ent_loss = entropy_loss(msfa, temp=temp, prior_scale=pscale)
                            esc_loss = escape_from_seed_loss(msfa, temp=temp, prior_scale=pscale, epoch=epoch)
                            if teacher_bundle is not None:
                                with torch.no_grad():
                                    teacher_pred = teacher_bundle["model"](teacher_bundle["msfa"](x))
                                distill_loss = F.smooth_l1_loss(pred, teacher_pred)
                            loss = (
                                loss
                                + LAMBDA_SP * sp_loss
                                + rc_w * rc_loss
                                + d_w * delta_loss
                                + ent_w * ent_loss
                                + LAMBDA_ESCAPE * esc_loss
                                + distill_weight(epoch) * distill_loss
                            )

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                        train_psnr += compute_psnr(pred, x).item()

                    model.eval()
                    msfa.eval()
                    val_psnr = 0.0
                    val_sam = 0.0
                    with torch.no_grad():
                        for x in val_loader:
                            x = x.to(DEVICE)
                            sensed = msfa(x) if mode_name == "osp" else msfa(x, temp=temp, prior_scale=pscale)
                            pred = model(sensed)
                            val_psnr += compute_psnr(pred, x).item()
                            val_sam += spectral_angle_mapper(pred, x).item()

                    train_loss /= len(train_loader)
                    train_psnr /= len(train_loader)
                    val_psnr /= len(val_loader)
                    val_sam /= len(val_loader)
                    quality = val_psnr - CKPT_SCORE_SAM_WEIGHT * val_sam
                    train_loss_list.append(train_loss)
                    val_psnr_list.append(val_psnr)

                    if mode_name == "learnable":
                        stats = distance_stats(msfa, temp=temp, prior_scale=pscale)
                        seed_value = seed_match(msfa, temp=temp, prior_scale=pscale).item()
                        soft_changed = 1.0 - seed_value
                        hard_changed = changed_fraction(msfa, temp=temp, prior_scale=pscale)
                    else:
                        stats = {"min_3d_distance": float("nan"), "mean_3d_distance": float("nan")}
                        seed_value = 1.0
                        soft_changed = 0.0
                        hard_changed = 0.0

                    history.append(
                        {
                            "epoch": epoch,
                            "mode": mode_name,
                            "train_loss": train_loss,
                            "train_psnr": train_psnr,
                            "val_psnr": val_psnr,
                            "val_sam_deg": val_sam,
                            "temperature": temp if mode_name == "learnable" else 1.0,
                            "prior_scale": pscale if mode_name == "learnable" else 1.0,
                            "sp_loss": float(sp_loss.item()) if mode_name == "learnable" else 0.0,
                            "rowcol_loss": float(rc_loss.item()) if mode_name == "learnable" else 0.0,
                            "delta_reg": float(delta_loss.item()) if mode_name == "learnable" else 0.0,
                            "entropy_loss": float(ent_loss.item()) if mode_name == "learnable" else 0.0,
                            "distill_loss": float(distill_loss.item()) if mode_name == "learnable" else 0.0,
                            "seed_match": seed_value,
                            "soft_changed_fraction": soft_changed,
                            "hard_changed_fraction": hard_changed,
                            "min_3d_distance": stats["min_3d_distance"],
                            "mean_3d_distance": stats["mean_3d_distance"],
                        }
                    )

                    if mode_name == "learnable":
                        print(
                            f"[learnable] Epoch {epoch:02d} | train PSNR {train_psnr:.2f} dB | "
                            f"val PSNR {val_psnr:.2f} dB | val SAM {val_sam:.2f} deg | "
                            f"seed {seed_value:.3f} | softchg {soft_changed:.3f} | hardchg {hard_changed:.3f}"
                        )
                    else:
                        print(
                            f"[osp] Epoch {epoch:02d} | train PSNR {train_psnr:.2f} dB | "
                            f"val PSNR {val_psnr:.2f} dB | val SAM {val_sam:.2f} deg"
                        )
                    scheduler.step()

                    if epoch % 5 == 0:
                        save_epoch_tile(msfa, mode_name, epoch, temp=temp, pscale=pscale)

                    if quality > best_quality:
                        best_psnr = val_psnr
                        best_quality = quality
                        epochs_since_improve = 0
                        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                        payload = {
                            "epoch": epoch,
                            "mode": mode_name,
                            "unet": model.state_dict(),
                            "best_val_psnr": best_psnr,
                            "best_quality": best_quality,
                            "warm_start_source": warm_start_source,
                            "init_ab": init_ab if init_tile_for_mode is None else "exact_selector_seed",
                            "init_source": init_source,
                            "hard_tile": (msfa.hard_tile(temp=temp, prior_scale=pscale) if mode_name == "learnable" else msfa.hard_tile()).detach().cpu().numpy(),
                        }
                            if mode_name == "learnable":
                                payload["msfa"] = msfa.state_dict()
                                payload["temperature"] = temp
                                payload["prior_scale"] = pscale
                                payload["seed_match"] = seed_value
                                payload["soft_changed_fraction"] = soft_changed
                                TILE_LEARN_SOFT_PATH.parent.mkdir(parents=True, exist_ok=True)
                                np.save(TILE_LEARN_SOFT_PATH, msfa.soft_tile(temp=temp, prior_scale=pscale).detach().cpu().numpy())
                            torch.save(payload, ckpt_path)
                    else:
                        epochs_since_improve += 1
                        if epochs_since_improve >= EARLY_STOP_PATIENCE:
                            print(f"[{mode_name}] Early stopping at epoch {epoch:02d}")
                            break

                history_path.parent.mkdir(parents=True, exist_ok=True)
                with open(history_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
                    writer.writeheader()
                    writer.writerows(history)

                return {
                    "mode": mode_name,
                    "history_path": str(history_path),
                    "ckpt_path": str(ckpt_path),
                    "best_val_psnr": best_psnr,
                    "train_loss_list": train_loss_list,
                    "val_psnr_list": val_psnr_list,
                    "model_state": model.state_dict(),
                    "hard_tile": (msfa.hard_tile(temp=TEMP_END, prior_scale=PRIOR_SCALE_END) if mode_name == "learnable" else msfa.hard_tile()).detach().cpu().numpy(),
                }

            run_results = {}
            exact_bundle = None
            if MODE in ["osp", "both"]:
                run_results["osp"] = train_mode("osp")
            if MODE in ["learnable", "both"]:
                exact_bundle = train_exact_selector(model_state=run_results.get("osp", {}).get("model_state"))
                run_results["exact_selector"] = exact_bundle
                teacher_model = UNet2D().to(DEVICE)
                teacher_model.load_state_dict(exact_bundle["model_state"], strict=True)
                teacher_model.eval()
                teacher_msfa = FixedOSP(init_tile_one_based=exact_bundle["selected_tile"]).to(DEVICE)
                teacher_msfa.eval()
                run_results["learnable"] = train_mode(
                    "learnable",
                    init_tile_for_mode=exact_bundle["selected_tile"],
                    warm_model_state=exact_bundle["model_state"],
                    teacher_bundle={"model": teacher_model, "msfa": teacher_msfa},
                )
            print("Completed modes:", list(run_results.keys()))
            """
        ),
        code_cell(
            """
            required_symbols = [
                "torch",
                "np",
                "plt",
                "DEVICE",
                "val_target",
                "val_loader",
                "UNet2D",
                "FixedOSP",
                "OSPSeededLearnableMSFA",
                "evaluate",
                "init_tile_one_based",
                "CKPT_OSP_PATH",
                "CKPT_EXACT_PATH",
                "CKPT_LEARN_PATH",
                "TEMP_END",
                "PRIOR_SCALE_END",
                "FIG_DIR",
            ]
            missing = [name for name in required_symbols if name not in globals()]
            if missing:
                raise RuntimeError(
                    "Notebook 11 final cell is being run without required setup cells. "
                    "Run cells 3 to 6 first. Missing: " + ", ".join(missing)
                )

            eval_rows = []
            loaded = {}

            if CKPT_OSP_PATH.exists():
                osp_ckpt = torch.load(CKPT_OSP_PATH, map_location=DEVICE, weights_only=False)
                osp_model = UNet2D().to(DEVICE)
                osp_model.load_state_dict(osp_ckpt["unet"], strict=True)
                osp_msfa = FixedOSP(init_tile_one_based=init_tile_one_based).to(DEVICE)
                psnr, sam, ssim_val = evaluate(osp_msfa, osp_model, val_loader, mode_name="osp")
                eval_rows.append({"Method": "OSP", "PSNR(dB)": psnr, "SAM(deg)": sam, "RGB-SSIM": ssim_val})
                loaded["osp"] = {"ckpt": osp_ckpt, "model": osp_model, "msfa": osp_msfa}

            if CKPT_EXACT_PATH.exists():
                exact_ckpt = torch.load(CKPT_EXACT_PATH, map_location=DEVICE, weights_only=False)
                exact_model = UNet2D().to(DEVICE)
                exact_model.load_state_dict(exact_ckpt["model"], strict=True)
                exact_msfa = FixedOSP(init_tile_one_based=torch.tensor(exact_ckpt["hard_tile"], dtype=torch.int64)).to(DEVICE)
                psnr, sam, ssim_val = evaluate(exact_msfa, exact_model, val_loader, mode_name="osp")
                eval_rows.append({"Method": "Exact OSP", "PSNR(dB)": psnr, "SAM(deg)": sam, "RGB-SSIM": ssim_val})
                loaded["exact"] = {"ckpt": exact_ckpt, "model": exact_model, "msfa": exact_msfa}

            if CKPT_LEARN_PATH.exists():
                lrn_ckpt = torch.load(CKPT_LEARN_PATH, map_location=DEVICE, weights_only=False)
                lrn_model = UNet2D().to(DEVICE)
                lrn_model.load_state_dict(lrn_ckpt["unet"], strict=True)
                learn_init_tile = torch.tensor(lrn_ckpt["hard_tile"], dtype=torch.int64)
                lrn_msfa = OSPSeededLearnableMSFA(init_tile_one_based=learn_init_tile).to(DEVICE)
                lrn_msfa.load_state_dict(lrn_ckpt["msfa"], strict=True)
                lrn_temp = lrn_ckpt.get("temperature", TEMP_END)
                lrn_prior = lrn_ckpt.get("prior_scale", PRIOR_SCALE_END)
                psnr, sam, ssim_val = evaluate(lrn_msfa, lrn_model, val_loader, mode_name="learnable", temp=lrn_temp, prior_scale=lrn_prior)
                eval_rows.append({"Method": "Learnable", "PSNR(dB)": psnr, "SAM(deg)": sam, "RGB-SSIM": ssim_val})
                loaded["learnable"] = {"ckpt": lrn_ckpt, "model": lrn_model, "msfa": lrn_msfa, "temp": lrn_temp, "prior": lrn_prior}

            if len(eval_rows) == 0:
                print("No checkpoints found. Run training first.")
            else:
                print("Final Results")
                print("-" * 64)
                print(f"{'Method':<12} {'PSNR(dB)':>10} {'SAM(deg)':>10} {'RGB-SSIM':>10}")
                print("-" * 64)
                for r in eval_rows:
                    print(f"{r['Method']:<12} {r['PSNR(dB)']:>10.3f} {r['SAM(deg)']:>10.3f} {r['RGB-SSIM']:>10.4f}")
                print("-" * 64)

            if "osp" in loaded and "exact" in loaded and "learnable" in loaded:
                FIG_DIR.mkdir(parents=True, exist_ok=True)
                osp_tile = loaded["osp"]["ckpt"]["hard_tile"]
                exact_tile = loaded["exact"]["ckpt"]["hard_tile"]
                lrn_tile = loaded["learnable"]["ckpt"]["hard_tile"]

                plt.figure(figsize=(15, 4))
                plt.subplot(1, 3, 1)
                plt.title("OSP")
                plt.imshow(osp_tile, cmap="tab20")
                plt.colorbar()
                plt.subplot(1, 3, 2)
                plt.title("Exact OSP")
                plt.imshow(exact_tile, cmap="tab20")
                plt.colorbar()
                plt.subplot(1, 3, 3)
                plt.title("Learned")
                plt.imshow(lrn_tile, cmap="tab20")
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(FIG_DIR / "phase11_pattern_comparison.png", dpi=200, bbox_inches="tight")
                plt.show()

                x = torch.from_numpy(val_target[0]).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
                with torch.no_grad():
                    osp_pred = loaded["osp"]["model"](loaded["osp"]["msfa"](x)).cpu().numpy()[0]
                    exact_pred = loaded["exact"]["model"](loaded["exact"]["msfa"](x)).cpu().numpy()[0]
                    lrn_pred = loaded["learnable"]["model"](
                        loaded["learnable"]["msfa"](x, temp=loaded["learnable"]["temp"], prior_scale=loaded["learnable"]["prior"])
                    ).cpu().numpy()[0]
                gt = x.cpu().numpy()[0]

                gt_rgb = np.transpose(gt[[5, 10, 15]], (1, 2, 0))
                osp_rgb = np.transpose(osp_pred[[5, 10, 15]], (1, 2, 0))
                exact_rgb = np.transpose(exact_pred[[5, 10, 15]], (1, 2, 0))
                lrn_rgb = np.transpose(lrn_pred[[5, 10, 15]], (1, 2, 0))

                def norm01(im):
                    mn = im.min()
                    mx = im.max()
                    return (im - mn) / max(mx - mn, 1e-8)

                gt_rgb = norm01(gt_rgb)
                osp_rgb = norm01(osp_rgb)
                exact_rgb = norm01(exact_rgb)
                lrn_rgb = norm01(lrn_rgb)
                osp_err = np.abs(gt - osp_pred).mean(axis=0)
                exact_err = np.abs(gt - exact_pred).mean(axis=0)
                lrn_err = np.abs(gt - lrn_pred).mean(axis=0)

                plt.figure(figsize=(18, 6))
                plt.subplot(2, 4, 1); plt.title("GT"); plt.imshow(gt_rgb); plt.axis("off")
                plt.subplot(2, 4, 2); plt.title("OSP Recon"); plt.imshow(osp_rgb); plt.axis("off")
                plt.subplot(2, 4, 3); plt.title("Exact OSP Recon"); plt.imshow(exact_rgb); plt.axis("off")
                plt.subplot(2, 4, 4); plt.title("Learned Recon"); plt.imshow(lrn_rgb); plt.axis("off")
                plt.subplot(2, 4, 6); plt.title("OSP Error"); plt.imshow(osp_err, cmap="hot"); plt.axis("off")
                plt.subplot(2, 4, 7); plt.title("Exact OSP Error"); plt.imshow(exact_err, cmap="hot"); plt.axis("off")
                plt.subplot(2, 4, 8); plt.title("Learned Error"); plt.imshow(lrn_err, cmap="hot"); plt.axis("off")
                plt.tight_layout()
                plt.savefig(FIG_DIR / "phase11_recon_and_error.png", dpi=200, bbox_inches="tight")
                plt.show()

                osp_row = next(r for r in eval_rows if r["Method"] == "OSP")
                exact_row = next(r for r in eval_rows if r["Method"] == "Exact OSP")
                lrn_row = next(r for r in eval_rows if r["Method"] == "Learnable")
                if (lrn_row["PSNR(dB)"] >= exact_row["PSNR(dB)"]) and (lrn_row["SAM(deg)"] <= exact_row["SAM(deg)"]):
                    print("Observation: Learnable outperforms both fixed and exact OSP branches in this setup.")
                elif (exact_row["PSNR(dB)"] >= osp_row["PSNR(dB)"]) and (exact_row["SAM(deg)"] <= osp_row["SAM(deg)"]):
                    print("Observation: Exact OSP selection improves the fixed OSP baseline, but learnable remains competitive.")
                else:
                    print("Observation: OSP-family methods remain competitive in this setup.")
            """
        ),
        md_cell(
            """
            This notebook now provides the full fair-comparison workflow:
            OSP vs learnable with shared loop, logging, evaluation, visualization, and final observations.
            """
        ),
    ]
)

notebooks["06_phase6_learned_method_figures.ipynb"] = notebook(
    [
        md_cell(
            """
            # Phase 6: Learned-Method Figures

            Goal: create separate qualitative figures for the learned method without overwriting the original OSP MATLAB figures.
            """
        ),
        md_cell(common_header),
        colab_mount_cell(),
        code_cell(
            """
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np

            PROJECT_ROOT = Path("/content/drive/MyDrive/Msa-Osp")
            TILE_PATH = PROJECT_ROOT / "learned_msfa_sp_tile_soft.npy"
            FIG_PATH = PROJECT_ROOT / "learned_msfa_sp_figure.png"

            BAND_COUNT = 16
            WAVELENGTHS = np.linspace(400.0, 700.0, BAND_COUNT, dtype=np.float32)
            """
        ),
        code_cell(
            """
            soft_tile = np.load(TILE_PATH).astype(np.float32)
            hard_tile_zero_based = soft_tile.argmax(axis=0).astype(np.int32)
            hard_tile_one_based = hard_tile_zero_based + 1
            centroid_nm = (soft_tile * WAVELENGTHS[:, None, None]).sum(axis=0).astype(np.float32)
            """
        ),
        code_cell(
            """
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(hard_tile_one_based, cmap="tab20")
            plt.title("Learned hard tile (1-based)")
            plt.colorbar()

            plt.subplot(1, 2, 2)
            plt.imshow(centroid_nm, cmap="viridis")
            plt.title("Centroid wavelengths")
            plt.colorbar(label="nm")
            plt.tight_layout()
            FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
            plt.show()
            """
        ),
        md_cell(
            """
            Use this figure separately from the original MATLAB OSP figures.

            It is the qualitative presentation of the learned method, not a replacement for the original discrete OSP figure.
            """
        ),
    ]
)

notebooks["07_phase7_export_to_matlab.ipynb"] = notebook(
    [
        md_cell(
            """
            # Phase 7: Optional MATLAB Export

            Goal: export the learned artifact for later MATLAB or hardware integration.

            This is a deployment notebook, not the main scientific result.
            """
        ),
        md_cell(common_header),
        colab_mount_cell(),
        code_cell(
            """
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            from scipy.io import savemat

            PROJECT_ROOT = Path("/content/drive/MyDrive/Msa-Osp")
            TILE_PATH = PROJECT_ROOT / "learned_msfa_sp_tile_soft.npy"
            MAT_PATH = PROJECT_ROOT / "learned_msfa_for_matlab.mat"

            BAND_COUNT = 16
            WAVELENGTHS = np.linspace(400.0, 700.0, BAND_COUNT, dtype=np.float32)
            """
        ),
        code_cell(
            """
            soft_tile = np.load(TILE_PATH).astype(np.float32)
            hard_tile_zero_based = soft_tile.argmax(axis=0).astype(np.int32)
            hard_tile_one_based = hard_tile_zero_based + 1
            centroid_nm = (soft_tile * WAVELENGTHS[:, None, None]).sum(axis=0).astype(np.float32)

            MAT_PATH.parent.mkdir(parents=True, exist_ok=True)
            savemat(
                MAT_PATH,
                {
                    "soft_tile": soft_tile,
                    "hard_tile_zero_based": hard_tile_zero_based,
                    "hard_tile_one_based": hard_tile_one_based,
                    "centroid_nm": centroid_nm,
                    "wavelengths_nm": WAVELENGTHS,
                },
            )
            print("Saved:", MAT_PATH)
            """
        ),
        code_cell(
            """
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(hard_tile_one_based, cmap="tab20")
            plt.title("Export hard tile (1-based)")
            plt.colorbar()

            plt.subplot(1, 2, 2)
            plt.imshow(centroid_nm, cmap="viridis")
            plt.title("Centroid wavelengths")
            plt.colorbar(label="nm")
            plt.tight_layout()
            plt.show()
            """
        ),
    ]
)

for name, content in notebooks.items():
    path = ROOT / name
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")
    print(f"Wrote {path.name}")
