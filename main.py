import argparse
from generation.bart.generation_task import generate_title_overview_recommendations as generate_bart
from generation.gpt2.generation_task import generate_title_overview_recommendations as generate_gpt2
from utils.data_loader import load_data
from utils.data_saver import save_output_data



def main():
    parser = argparse.ArgumentParser(description="Generate title, overview, and recommendations using BART, BERT, or GPT-2.")
    parser.add_argument('--model', type=str, required=True, choices=['bart', 'bert', 'gpt2'], help="Model type to use for generation.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input data JSON file.")
    parser.add_argument('--config', type=str, required=True, help="Path to the config JSON file.")

    args = parser.parse_args()
    config = load_data(args.config)
    model_path = config['generation']['model_paths'][args.model]
    output_path = config['generation']['output_path'][args.model]

    input_data = load_data(args.input)

    if args.model == 'bart':
        title, overview, recommendations = generate_bart(input_data, model_path)
    elif args.model == 'gpt2':
        title, overview, recommendations = generate_gpt2(input_data, model_path)

    output_data = {
        'title': title,
        'overview': overview,
        'recommendations': recommendations
    }

    save_output_data(output_data, output_path)
    print("Generation complete. Output saved to", output_path)

if __name__ == "__main__":
    main()
