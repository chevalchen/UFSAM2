import argparse

def get_args_parser() -> argparse.ArgumentParser:
    """Argument parser for SANSA training and inference (parent parser)."""
    parser = argparse.ArgumentParser("SANSA training and inference", add_help=False)

    # General
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device.")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from.")

    # Experiment I/O
    parser.add_argument("--output_dir", type=str, default="output", help="Root directory for outputs.")
    parser.add_argument("--name_exp", type=str, default="prova", help="Experiment name (subfolder in output_dir).")

    # Data
    parser.add_argument("--data_root", type=str, default="data", help="Root directory for datasets.")
    parser.add_argument("--dataset_file", type=str, default="coco", choices=["coco", "lvis", "fss", "pascal_voc", "pascal_voc_cd", "pascal_part", "paco_part", "deepglobe", "isic",
                 "lung", "ade20k", "multi"], help="Dataset name. Use 'multi' for training the generalist model.")
    parser.add_argument("--multi_train", nargs="+", type=str, default=["lvis", "coco", "ade20k", "paco_part"], help="Datasets to mix when dataset_file='multi'.")
    parser.add_argument("--ds_weight", nargs="+", type=float, default=[0.4, 0.45, 0.1, 0.05], help="Sampling weights for datasets in --multi_train.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers.")

    # Prompting / Shots / Folds
    parser.add_argument("--prompt", type=str, default="mask", choices=["mask", "scribble", "box", "point", "multi"], help="Prompt type for support frames; 'multi' samples a type at random.")
    parser.add_argument("--shots", type=int, default=1, help="Number of support frames per episode.")
    parser.add_argument("--J", type=int, default=3, help="Number of unlabeled target images per training episode.")
    parser.add_argument("--fold", type=int, default=0, help="Training fold (if applicable).")

    # SAM2 / Backbone
    parser.add_argument("--sam2_version", type=str, default="large", choices=["tiny", "base", "large"], help="Version of SAM2 image encoder.")
    parser.add_argument("--adaptformer_stages", nargs="+", type=int, default=[-1], help="Adapter/AdaptFormer stages to enable (model-specific).")
    parser.add_argument("--channel_factor", type=float, default=0.3, help="Adapter channel scaling factor (model-specific).")

    # Optimization
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument('--start_epoch', default=0, type=int, help="Epoch to start training from (for resuming)")
    parser.add_argument("--clip_max_norm", type=float, default=0.1, help="Gradient clipping max norm (0 disables clipping).")
    parser.add_argument("--batch_size", type=int, default=2, help="Global batch size (may be split per GPU).")

    # Logging / Runtime
    parser.add_argument("--no_distributed", action="store_true", default=False, help="Force single-process training.")

    # Inference
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold to binarize masks at eval.")
    parser.add_argument("--visualize", action="store_true", default=False, help="Save qualitative results.")
    parser.add_argument("--max_eval_episodes", type=int, default=None, help="Optional cap for quick validation; omit for full official evaluation.")
    parser.add_argument("--hflip_tta", action="store_true", default=False, help="Average query logits with horizontal-flip TTA.")
    parser.add_argument("--uq_hflip_tta", action="store_true", default=False, help="Trigger horizontal-flip TTA only for low-confidence query episodes.")
    parser.add_argument("--boundary_refine", action="store_true", default=False, help="Apply the trainable boundary refinement module after query decoding.")
    parser.add_argument("--memory_to_point_prompt", action="store_true", default=False, help="Run uncertainty-triggered memory-to-point self-prompting on query frames.")
    parser.add_argument("--mtp_trigger_threshold", type=float, default=0.92, help="Trigger memory-to-point prompting when the unsupervised query quality score is below this value.")
    parser.add_argument("--mtp_accept_margin", type=float, default=0.0, help="Accept the memory-to-point second pass only if its quality improves by at least this margin.")
    parser.add_argument("--mtp_num_positive_points", type=int, default=1, help="Number of automatic positive points sampled from confident query foreground.")
    parser.add_argument("--mtp_num_negative_points", type=int, default=1, help="Number of automatic negative points sampled from query disagreement/background.")
    parser.add_argument("--mtp_pos_threshold", type=float, default=0.65, help="Minimum foreground probability for automatic positive points.")
    parser.add_argument("--mtp_neg_threshold", type=float, default=0.35, help="Maximum selected-mask foreground probability for automatic negative points.")
    parser.add_argument("--uq_head_ckpt", type=str, default="", help="Expected-IoU head checkpoint for uncertainty-gated hflip TTA.")
    parser.add_argument("--uq_head_device", type=str, default="cpu", help="Device for the expected-IoU head used by --uq_hflip_tta.")
    parser.add_argument("--uq_gate_threshold", type=float, default=0.5, help="Run gated hflip when predicted expected IoU is below this threshold.")
    parser.add_argument("--support_agg", type=str, default="none", choices=["none", "weighted_logits"], help="5-shot support aggregation strategy for Module B.")
    parser.add_argument("--support_uq_head_ckpt", type=str, default="", help="Expected-IoU head checkpoint for uncertainty-guided support aggregation.")
    parser.add_argument("--support_uq_head_device", type=str, default="cpu", help="Device for the expected-IoU head used by --support_agg.")
    parser.add_argument("--support_weight_temp", type=float, default=0.0, help="Softmax temperature for support scores; <=0 uses linear score normalization.")
    parser.add_argument("--support_fallback_margin", type=float, default=0.03, help="Fallback to all-support SANSA when support scores are nearly tied.")
    parser.add_argument("--support_fallback_min_score", type=float, default=0.0, help="Fallback to all-support SANSA when every support score is below this value.")
    parser.add_argument("--post_memory_calibration", action="store_true", default=False, help="Enable AV-PMC post-memory feature calibration on query frames.")
    parser.add_argument("--pmc_mode", type=str, default="gated", choices=["operator", "spatial", "gated"], help="AV-PMC path: all-feature operator, spatial gate, or spatial plus episode action-value gate.")
    parser.add_argument("--pmc_checkpoint", type=str, default="", help="Checkpoint containing PostMemoryFeatureCalibrator weights.")
    parser.add_argument("--pmc_projection_dim", type=int, default=64, help="Projected width for pre/post-memory disagreement features.")
    parser.add_argument("--pmc_hidden_dim", type=int, default=128, help="Hidden width of the AV-PMC dense evidence trunk.")
    parser.add_argument("--pmc_residual_scale", type=float, default=0.1, help="Maximum magnitude scale of the bounded feature residual.")
    parser.add_argument("--pmc_spatial_threshold", type=float, default=0.0, help="Signed dense-benefit threshold used by the spatial gate.")
    parser.add_argument("--pmc_episode_threshold", type=float, default=0.0, help="Predicted delta-IoU threshold for running the second decoder pass.")
    parser.add_argument("--pmc_gate_temperature", type=float, default=0.25, help="Temperature of the soft spatial benefit gate.")
    parser.add_argument("--pmc_train_stage", type=str, default=None, choices=["operator", "spatial", "gain"], help="Train exactly one AV-PMC stage while freezing SANSA; used by the dedicated training script.")

    return parser
