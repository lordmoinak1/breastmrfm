import os
import torch
import argparse
import numpy as np
from tqdm import tqdm

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    NormalizeIntensityd, ResizeWithPadOrCropd, ToTensord
)
from monai.data import Dataset, DataLoader, pad_list_data_collate

from simclr import SimCLRModule  # Adjust if using a different filename


# --------- Inference Transform ---------
def get_infer_loader(data_dir, batch_size=1):
    infer_transforms = Compose([
        LoadImaged(keys=['dce']),
        EnsureChannelFirstd(keys=['dce']),
        Orientationd(keys=['dce'], axcodes="RAS"),
        Spacingd(keys=['dce'], pixdim=(2.0, 2.0, 2.0), mode="bilinear"),
        NormalizeIntensityd(keys=['dce'], nonzero=True, channel_wise=True),
        ResizeWithPadOrCropd(keys=['dce'], spatial_size=(128, 128, 128)),
        ToTensord(keys=['dce'])
    ])

    def get_subjects(data_path):
        subjects = []
        for fname in os.listdir(data_path):
            if fname.endswith(".nii.gz"):
                subj_id = fname.replace(".nii.gz", "")
                subj = {
                    'dce': os.path.join(data_path, fname),
                    'id': subj_id
                }
                subjects.append(subj)
        return subjects

    subjects = get_subjects(data_dir)
    dataset = Dataset(data=subjects, transform=infer_transforms)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, collate_fn=pad_list_data_collate)
    return loader, subjects


# --------- Run Inference ---------
def run_inference_simclr(ckpt_path, data_dir, output_path):
    model = SimCLRModule.load_from_checkpoint(ckpt_path, strict=False)
    model.eval().cuda()

    loader, subjects = get_infer_loader(data_dir)

    os.makedirs(output_path, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            subj = subjects[i]
            x = batch["dce"].cuda()

            features = model.encoder(x)  # Get raw features before projection head
            emb = features[0].cpu().numpy()

            np.save(os.path.join(output_path, f"{subj['id']}.npy"), emb)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SimCLR inference to extract features."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the checkpoint file (.ckpt)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the input dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save extracted features"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    run_inference_simclr(
        ckpt_path=args.ckpt_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

  # python3 generate_features.py --ckpt_path /path/to/checkpoints/epoch=99-step=46200.ckpt --data_dir /path/to/test/duke --output_dir /path/to/features/duke
