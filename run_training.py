import argparse
import logging
import os
from fine_tuning.finetuning import fine_tune_model
from utils.data_loader import load_data
from scripts.track_training import track_training_progress
from scripts.resume_training import resume_training

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Fine tune the model BART or GPT-2.")
    parser.add_argument('--model', type=str, required=True, choices=['bart', 'gpt2'], help="Model type to use for fine tuning.")
    parser.add_argument('--dataType', type=str, required=True, choices=['cybersecurity', 'original'], help="Data type to use for fine tuning.")
    parser.add_argument('--config', type=str, required=True, help="Path to the config JSON file.")
    parser.add_argument('--resume', action='store_true', help="Resume training from the last checkpoint.")
    parser.add_argument('--track', action='store_true', help="Track training progress.")

    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging
    log_file = os.path.basename(__file__).replace(".py", ".log")
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    if args.track:
        logging.info("Starting training tracking...")
        track_training_progress()
        return

    if args.resume:
        logging.info("Resuming training...")
        resume_training(training_type=args.model, data_type=args.dataType, config_file=args.config)
        return

    # Load configuration file
    logging.info("Loading configuration file...")
    config = load_data(args.config)
    
    data_type = args.dataType
    logging.info(f"Selected model: {args.model}")
    logging.info(f"Selected data type: {args.dataType}")

    # Fine tune the selected model with the specified data type
    try:
        logging.info(f"Start fine tuning ...")

        fine_tune_model(args.model, args.dataType, config)

        logging.info("Fine tuning finished successfully.")
    except Exception as e:
        logging.error(f"An error occurred during fine tuning: {e}")

if __name__ == "__main__":
    main()
