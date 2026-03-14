"""Evaluate DigitCNN with per-digit and per-frame metrics.

This evaluation matches the current architecture:
- Digit-level model predictions (single digit classes)
- Per-position confusion matrices (slot indices 0..3)
- Per-frame reconstruction into x.xxx strings
"""

import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import FILTER_CONF_THRESH, FILTER_ENT_THRESH
from core.dataset import ScaleDigitDataset, digit_collate_fn, get_transforms
from core.model import create_model


def should_flag(prediction, confidence, entropy, conf_thresh, ent_thresh):
    """Flag predictions with invalid format or weak confidence/entropy."""
    pattern = re.compile(r"^\d\.\d{3}$")

    if not pattern.match(prediction):
        return True, "bad_format"

    try:
        float(prediction)
    except ValueError:
        return True, "parse_error"

    if confidence == 0.0:
        return True, "low_nonblank"

    if confidence < conf_thresh:
        return True, "low_conf"

    if entropy > ent_thresh:
        return True, "high_ent"

    return False, None


def _load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        state_dict = {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}

    model = create_model(device=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _predict_digits_batch(model, images, device):
    images = images.to(device)
    logits = model(images)
    probs = torch.softmax(logits, dim=1)

    confs, preds = probs.max(dim=1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)

    return preds.cpu(), confs.cpu(), entropy.cpu()


def _assemble_weight_from_digits(d0, d1, d2, d3):
    return f"{d0}.{d1}{d2}{d3}"


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from: {args.model}")
    try:
        model = _load_model(args.model, device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    transforms = get_transforms(is_train=False)
    try:
        dataset = ScaleDigitDataset(
            labels_csv=args.labels,
            images_dir=args.images,
            transform=transforms,
            validate=True,
            verbose=False
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=digit_collate_fn
    )

    print(f"Evaluating on {len(dataset)} digit samples from: {args.labels}")
    print(f"Main report will be saved to: {args.output}")

    slot_conf_mats = {slot: np.zeros((10, 10), dtype=np.int64) for slot in range(4)}
    slot_total = {slot: 0 for slot in range(4)}
    slot_correct = {slot: 0 for slot in range(4)}

    digit_rows = []
    frame_slots = defaultdict(dict)
    frame_has_duplicate = defaultdict(bool)

    total_digits = 0
    total_correct_digits = 0

    with torch.no_grad():
        for images, labels, frame_ids, slot_idxs in tqdm(dataloader, desc="Running Inference"):
            preds, confs, ents = _predict_digits_batch(model, images, device)

            for i in range(labels.shape[0]):
                gt_digit = int(labels[i].item())
                pred_digit = int(preds[i].item())
                conf = float(confs[i].item())
                ent = float(ents[i].item())
                frame_id = frame_ids[i]
                slot_idx = int(slot_idxs[i].item())
                is_correct = pred_digit == gt_digit

                total_digits += 1
                total_correct_digits += int(is_correct)

                if slot_idx in slot_conf_mats:
                    slot_conf_mats[slot_idx][gt_digit, pred_digit] += 1
                    slot_total[slot_idx] += 1
                    slot_correct[slot_idx] += int(is_correct)

                digit_rows.append(
                    {
                        "frame_id": frame_id,
                        "slot_index": slot_idx,
                        "ground_truth_digit": gt_digit,
                        "predicted_digit": pred_digit,
                        "is_correct": is_correct,
                        "confidence": conf,
                        "entropy": ent,
                    }
                )

                if slot_idx in frame_slots[frame_id]:
                    frame_has_duplicate[frame_id] = True
                else:
                    frame_slots[frame_id][slot_idx] = {
                        "gt": gt_digit,
                        "pred": pred_digit,
                        "conf": conf,
                        "ent": ent,
                    }

    frame_rows = []
    valid_frame_count = 0
    correct_frame_count = 0

    for frame_id, slots in frame_slots.items():
        duplicate_slots = frame_has_duplicate[frame_id]
        has_all_slots = all(slot in slots for slot in range(4))

        status = "complete"
        if not has_all_slots:
            status = "incomplete"
        elif duplicate_slots:
            status = "duplicate_slots"

        pred_weight = None
        gt_weight = None
        exact_match = False
        pred_numeric = np.nan
        gt_numeric = np.nan
        abs_error = np.nan
        agg_conf = 0.0
        agg_ent = float("inf")
        flagged = True
        flag_reason = "incomplete_slots"

        pred_slot_digits = [slots.get(slot, {}).get("pred", np.nan) for slot in range(4)]
        gt_slot_digits = [slots.get(slot, {}).get("gt", np.nan) for slot in range(4)]
        slot_confs = [slots.get(slot, {}).get("conf", np.nan) for slot in range(4)]
        slot_ents = [slots.get(slot, {}).get("ent", np.nan) for slot in range(4)]

        if has_all_slots:
            pred_weight = _assemble_weight_from_digits(
                slots[0]["pred"],
                slots[1]["pred"],
                slots[2]["pred"],
                slots[3]["pred"],
            )
            gt_weight = _assemble_weight_from_digits(
                slots[0]["gt"],
                slots[1]["gt"],
                slots[2]["gt"],
                slots[3]["gt"],
            )
            exact_match = pred_weight == gt_weight

            agg_conf = float(min(slots[s]["conf"] for s in range(4)))
            agg_ent = float(np.mean([slots[s]["ent"] for s in range(4)]))
            flagged, flag_reason = should_flag(
                pred_weight,
                agg_conf,
                agg_ent,
                args.confidence_threshold,
                args.entropy_threshold,
            )

            try:
                pred_numeric = float(pred_weight)
                gt_numeric = float(gt_weight)
                abs_error = abs(pred_numeric - gt_numeric)
            except (ValueError, TypeError):
                pred_numeric = np.nan
                gt_numeric = np.nan
                abs_error = np.nan

            if status == "complete":
                valid_frame_count += 1
                correct_frame_count += int(exact_match)

        if status == "duplicate_slots":
            flagged = True
            flag_reason = "duplicate_slots"

        frame_rows.append(
            {
                "frame_id": frame_id,
                "status": status,
                "ground_truth": gt_weight,
                "prediction": pred_weight,
                "exact_match": exact_match,
                "pred_numeric": pred_numeric,
                "gt_numeric": gt_numeric,
                "abs_error": abs_error,
                "confidence": agg_conf,
                "entropy": agg_ent,
                "was_flagged": flagged,
                "flag_reason": flag_reason,
                "gt_digit_0": gt_slot_digits[0],
                "gt_digit_1": gt_slot_digits[1],
                "gt_digit_2": gt_slot_digits[2],
                "gt_digit_3": gt_slot_digits[3],
                "pred_digit_0": pred_slot_digits[0],
                "pred_digit_1": pred_slot_digits[1],
                "pred_digit_2": pred_slot_digits[2],
                "pred_digit_3": pred_slot_digits[3],
                "digit_conf_0": slot_confs[0],
                "digit_conf_1": slot_confs[1],
                "digit_conf_2": slot_confs[2],
                "digit_conf_3": slot_confs[3],
                "digit_ent_0": slot_ents[0],
                "digit_ent_1": slot_ents[1],
                "digit_ent_2": slot_ents[2],
                "digit_ent_3": slot_ents[3],
            }
        )

    frame_df = pd.DataFrame(frame_rows)
    digit_df = pd.DataFrame(digit_rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_df.to_csv(output_path, index=False)

    digit_out = output_path.with_name(f"{output_path.stem}_digits.csv")
    digit_df.to_csv(digit_out, index=False)

    for slot in range(4):
        conf_mat_df = pd.DataFrame(
            slot_conf_mats[slot],
            index=[f"true_{d}" for d in range(10)],
            columns=[f"pred_{d}" for d in range(10)],
        )
        conf_out = output_path.with_name(f"{output_path.stem}_confmat_slot{slot}.csv")
        conf_mat_df.to_csv(conf_out)

    digit_accuracy = (total_correct_digits / total_digits * 100.0) if total_digits > 0 else 0.0
    frame_accuracy = (correct_frame_count / valid_frame_count * 100.0) if valid_frame_count > 0 else 0.0
    flagged_count = int(frame_df["was_flagged"].sum()) if len(frame_df) > 0 else 0
    flag_rate = (flagged_count / len(frame_df) * 100.0) if len(frame_df) > 0 else 0.0
    incomplete_count = int((frame_df["status"] != "complete").sum()) if len(frame_df) > 0 else 0

    print("\n" + "=" * 52)
    print("EVALUATION SUMMARY")
    print("=" * 52)
    print(f"Total Digit Samples:      {total_digits}")
    print(f"Digit Accuracy:           {digit_accuracy:.2f}% ({total_correct_digits}/{total_digits})")
    for slot in range(4):
        slot_acc = (slot_correct[slot] / slot_total[slot] * 100.0) if slot_total[slot] > 0 else 0.0
        print(f"Slot {slot} Accuracy:           {slot_acc:.2f}% ({slot_correct[slot]}/{slot_total[slot]})")
    print("-" * 52)
    print(f"Total Frames (grouped):   {len(frame_df)}")
    print(f"Valid Frames:             {valid_frame_count}")
    print(f"Frame Exact Match:        {frame_accuracy:.2f}% ({correct_frame_count}/{valid_frame_count})")
    print(f"Incomplete Frames:        {incomplete_count}")
    print(f"Flagged Rate:             {flag_rate:.2f}% ({flagged_count}/{len(frame_df)})")
    print("-" * 52)
    print(f"Frame report CSV:         {output_path}")
    print(f"Digit detail CSV:         {digit_out}")
    print("Confusion matrices:       <output_stem>_confmat_slot{0..3}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Scale OCR Digit Model")
    parser.add_argument("--model", type=str, default="data/models/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--labels", type=str, default=None, help="Path to labels CSV")
    parser.add_argument("--test", action="store_true", help="Evaluate on the test set")
    parser.add_argument("--images", type=str, default="data/images", help="Path to images directory")

    default_out = f"data/outputs/val_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    parser.add_argument("--output", type=str, default=default_out, help="Main output CSV path")

    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=FILTER_CONF_THRESH,
        help="Flagging confidence threshold",
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=FILTER_ENT_THRESH,
        help="Flagging entropy threshold",
    )

    args = parser.parse_args()

    if args.labels is None:
        if args.test:
            args.labels = "data/labels/test_labels.csv"
        else:
            args.labels = "data/labels/val_labels.csv"

    if args.output == default_out:
        set_name = "test" if args.test else "val"
        args.output = f"data/outputs/{set_name}_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    evaluate(args)
