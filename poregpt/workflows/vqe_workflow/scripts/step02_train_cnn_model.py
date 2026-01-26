# test/test_vq_tokenizer/test_train_vq_model.py
from nanopore_signal_tokenizer import cnn_train
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_dir", type=str, required=True)
    parser.add_argument("--output_model_path", type=str, default="demo_nanopore_vq_tokenizer.pth")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--chunk_size", type=int, default=12000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--loss_csv_path", type=str, default="train_loss.csv")
    parser.add_argument("--loss_log_interval", type=int, default=10)
    parser.add_argument("--do_evaluate", action="store_true", help="Enable codebook usage evaluation")
    parser.add_argument("--checkpoint_path", type=str, default="checkpiint_nanopore_vq_tokenizer.pth")
    parser.add_argument("--cnn_type", type=int, default=0)
    args = parser.parse_args()

    cnn_train(
        npy_dir=args.npy_dir,
        output_model_path=args.output_model_path,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        do_evaluate=args.do_evaluate,
        loss_csv_path=args.loss_csv_path,
        loss_log_interval=args.loss_log_interval,
        checkpoint_path=args.checkpoint_path,
        cnn_type=args.cnn_type
    )

