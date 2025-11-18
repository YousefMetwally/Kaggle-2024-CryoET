import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.jit
from torch.utils.data import Dataset, DataLoader
import argparse
import mrcfile
from typing import List, Tuple, Union
from cryoet.modelling.detection.functional import decode_detections_with_nms
from cryoet.data.functional import normalize_volume_to_unit_range
from cryoet.inference.dataset import TileDataset
from cryoet.training.od_accumulator import AccumulatedObjectDetectionPredictionContainer
from pathlib import Path




TARGET_SIGMAS = [6]
CLASS_LABEL_TO_CLASS_NAME = {'0' : 'Particle'}




def get_volume_mrc(v_path):
    with mrcfile.open(v_path, permissive=True) as mrc:
        tomo_array = mrc.data.astype(np.float32)
    return(tomo_array)

def infer_num_classes_from_logits(logits):
    if not torch.is_tensor(logits):
        logits = logits[0]

    b, c, d, h, w = logits.size()
    return int(c)

@torch.no_grad()
@torch.jit.optimized_execution(False)
def predict_volume(
    volume: np.ndarray,
    models: List,
    output_strides: List[int],
    window_size: Tuple[int, int, int],
    tiles_per_dim: Tuple[int, int, int],
    device: str,
    study_name: str,
    score_thresholds: Union[float, List[float]],
    iou_threshold,
    batch_size,
    num_workers,
    use_weighted_average,
    use_centernet_nms,
    use_single_label_per_anchor,
    torch_dtype,
    pre_nms_top_k,
    use_z_flip_tta: bool,
    use_y_flip_tta: bool,
    use_x_flip_tta: bool,
    sigma: Union[int, Tuple[int, int, int]]
):
    scores, offsets = predict_scores_offsets_from_volume(
        volume=volume,
        models=models,
        output_strides=output_strides,
        window_size=window_size,
        tiles_per_dim=tiles_per_dim,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        torch_dtype=torch_dtype,
        study_name=study_name,
        use_weighted_average=use_weighted_average,
        use_z_flip_tta=use_z_flip_tta,
        use_y_flip_tta=use_y_flip_tta,
        use_x_flip_tta=use_x_flip_tta,
        sigma=sigma
    )
    submission = postprocess_scores_offsets_into_submission(
        scores=scores,
        offsets=offsets,
        iou_threshold=iou_threshold,
        output_strides=output_strides,
        score_thresholds=score_thresholds,
        study_name=study_name,
        use_centernet_nms=use_centernet_nms,
        use_single_label_per_anchor=use_single_label_per_anchor,
        pre_nms_top_k=pre_nms_top_k,
    )
    return submission


def postprocess_scores_offsets_into_submission(
    iou_threshold,
    offsets,
    output_strides,
    score_thresholds,
    scores,
    study_name,
    use_centernet_nms,
    use_single_label_per_anchor,
    pre_nms_top_k: int,
):
    topk_coords_px, topk_clses, topk_scores = decode_detections_with_nms(
        scores=scores,
        offsets=offsets,
        strides=output_strides,
        class_sigmas=TARGET_SIGMAS,
        min_score=score_thresholds,
        iou_threshold=iou_threshold,
        use_centernet_nms=use_centernet_nms,
        use_single_label_per_anchor=use_single_label_per_anchor,
        pre_nms_top_k=pre_nms_top_k,
    )
    topk_scores = topk_scores.float().cpu().numpy()
    top_coords = topk_coords_px.float().cpu().numpy() 
    topk_clses = topk_clses.cpu().numpy()
    submission = dict(
        experiment=[],
        particle_type=[],
        score=[],
        x=[],
        y=[],
        z=[],
    )
    for cls, coord, score in zip(topk_clses, top_coords, topk_scores):
        submission["experiment"].append(study_name)
        submission["particle_type"].append(CLASS_LABEL_TO_CLASS_NAME[int(cls)])
        submission["score"].append(float(score))
        submission["x"].append(float(coord[0]))
        submission["y"].append(float(coord[1]))
        submission["z"].append(float(coord[2]))
    submission = pd.DataFrame.from_dict(submission)
    return submission


@torch.no_grad()
def predict_scores_offsets_from_volume(
    batch_size,
    device,
    models,
    num_workers,
    output_strides,
    study_name,
    torch_dtype,
    use_weighted_average,
    volume,
    window_size: Tuple[int, int, int],
    tiles_per_dim: Tuple[int, int, int],
    use_z_flip_tta: bool,
    use_y_flip_tta: bool,
    use_x_flip_tta: bool,
    sigma: Union[int, Tuple[int, int, int]]
):
    torch.cuda.empty_cache()
    container = None
    volume = normalize_volume_to_unit_range(volume)
    ds = TileDataset(volume, window_size, sigma, torch_dtype=torch_dtype)
    for tile_volume, tile_offsets in tqdm(
        DataLoader(ds, batch_size=batch_size, num_workers=num_workers, drop_last=False, pin_memory=True),
        desc=f"{study_name} {volume.shape}",
    ):
        tile_volume = tile_volume.to(device=device, non_blocking=True)

        for model in models:
            probas, offsets = model(tile_volume)

            if torch.is_tensor(probas):
                probas = [probas]
            if torch.is_tensor(offsets):
                offsets = [offsets]

            if container is None:
                num_classes = infer_num_classes_from_logits(probas)
                print("Num classes", num_classes)

                container = AccumulatedObjectDetectionPredictionContainer.from_shape(
                    shape=volume.shape,
                    num_classes=num_classes,
                    window_size=window_size,
                    use_weighted_average=use_weighted_average,
                    strides=output_strides,
                    device=device,
                    dtype=torch_dtype,
                    sigma=sigma
                )
            #container.__post_init__()

            container.accumulate_batch(probas, offsets, tile_offsets)


    scores, offsets = container.merge_()
    return scores, offsets


def main():
    parser = argparse.ArgumentParser(description='This script will take an input tomogram, run the object detection and then return an output file with the same name as the tomogram in the output folder')
    parser.add_argument('--tomo_path', help='Path to the input tomo')
    parser.add_argument('--model', help='Path to the OD model')
    parser.add_argument('--out', help='Path to the output folder')

    args = parser.parse_args()

    model = torch.jit.load(args.model)
    model.eval()
    models = [model]

    volume = get_volume_mrc(args.tomo_path)
    tomo = os.path.basename(args.tomo_path)
    output_strides = output_strides = (2,)
    window_size = window_size = (256 , 296, 296)
    tiles_per_dim = (1,1,1) # this parameter is currently not used
    use_weighted_average = True
    use_centernet_nms = True
    use_single_label_per_anchor = False
    use_z_flip_tta = False
    use_y_flip_tta = False
    use_x_flip_tta = False
    batch_size = 1
    num_workers = 0
    device = 'cuda'
    torch_dtype = torch.float16

    study_sub = predict_volume(
            volume=volume,
            study_name=tomo,
            output_strides=output_strides,
            models= models,
            window_size=window_size,
            tiles_per_dim=tiles_per_dim,
            use_weighted_average=use_weighted_average,
            use_centernet_nms=use_centernet_nms,
            use_single_label_per_anchor=use_single_label_per_anchor,
            pre_nms_top_k=None,

            use_z_flip_tta=use_z_flip_tta,
            use_y_flip_tta=use_y_flip_tta,
            use_x_flip_tta=use_x_flip_tta,

            score_thresholds=[0.005],
            iou_threshold=0.6,
            sigma= (0,16,16),
            batch_size=batch_size,
            num_workers=num_workers,

            device=device,
            torch_dtype=torch_dtype,
        )
    
    corresponding_threshold = 0
    OUTPUT_DIR = Path(f"{args.out}/{tomo}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_PATH = f"{args.out}/{tomo}/{tomo}_submission.csv"

    study_sub[(study_sub['score']>corresponding_threshold)].to_csv(f"{CSV_PATH}", index=False)
    submission = pd.read_csv(CSV_PATH)
    submission[['x', 'y','z']] = submission[['x', 'y','z']]
    submission = submission[
    (submission['z'] > 0) & (submission['z'] < volume.shape[0] - 1) &
    (submission['y'] > 0) & (submission['y'] < volume.shape[1] - 1) &
    (submission['x'] > 0) & (submission['x'] < volume.shape[2] - 1)
]
    submission.to_csv(f"{CSV_PATH}", index=False)






