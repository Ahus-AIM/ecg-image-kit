import os
import random
import warnings
from scipy.stats import bernoulli
from tqdm import tqdm
from extract_leads import get_paper_ecg
from CreasesWrinkles.creases import get_creased
from helper_functions import read_config_file

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")


def run_single_file(configs):
    input_directory = configs["input_directory"]
    header_directory = configs["header_directory"]
    output_directory = configs["output_directory"]
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    wrinkles = configs.get("wrinkles", False)

    resolution = configs.get("resolution", 200)
    padding = configs.get("padding", 0)

    papersize = configs.get("papersize", "")
    lead = configs.get("lead", True)

    bernoulli_dc = bernoulli(configs.get("bernoulli_dc", 0.5))
    bernoulli_bw = bernoulli(configs.get("bernoulli_bw", 0))
    bernoulli_grid = bernoulli(configs.get("bernoulli_grid", 1))
    bernoulli_add_print = bernoulli(configs.get("bernoulli_add_print", 0))

    standard_colours = configs.get("standard_colours", False)

    # Read the start and end indices from the config
    start_index = configs["start_index"]
    end_index = configs["end_index"]

    # Dynamically generate input and header files based on the indices
    loading_bar = tqdm(range(start_index, end_index + 1), desc="Generating ECG Images")
    for i in loading_bar:
        folder = f"{i // 1000 * 1000:05d}"
        file_index = f"{i:05d}"

        input_file = os.path.join(input_directory, folder, f"{file_index}_lr.dat")
        header_file = os.path.join(header_directory, folder, f"{file_index}_lr.hea")

        out_array = get_paper_ecg(
            input_file=input_file,
            header_file=header_file,
            configs=configs,
            mask_unplotted_samples=configs.get("mask_unplotted_samples", False),
            store_configs=configs.get("store_configs", 0),
            store_text_bbox=configs.get("store_text_bbox", False),
            output_directory=output_directory,
            resolution=resolution,
            papersize=papersize,
            add_lead_names=lead,
            add_dc_pulse=bernoulli_dc,
            add_bw=bernoulli_bw,
            show_grid=bernoulli_grid,
            add_print=bernoulli_add_print,
            pad_inches=padding,
            font_type=None,
            standard_colours=standard_colours,
            full_mode=configs.get("full_mode", "II"),
            bbox=configs.get("bbox", False),
            columns=configs.get("columns", -1),
            seed=configs.get("seed", 42),
        )

        # Optionally apply wrinkles if needed
        wrinkles_config = configs.get("wrinkles_config", {})
        if wrinkles:
            for out in out_array:
                out = get_creased(
                    out,
                    output_directory=output_directory,
                    ifWrinkles=wrinkles_config.get("ifWrinkles", True),
                    ifCreases=wrinkles_config.get("ifCreases", True),
                    crease_angle=wrinkles_config.get(
                        "crease_angle", random.choice(range(0, 90))
                    ),
                    num_creases_vertically=wrinkles_config.get(
                        "num_creases_vertically", random.choice(range(1, 5))
                    ),
                    num_creases_horizontally=wrinkles_config.get(
                        "num_creases_horizontally", random.choice(range(1, 5))
                    ),
                    bbox=wrinkles_config.get("bbox", False),
                )
    return len(out_array)


if __name__ == "__main__":
    config_path = "src/config/config.yml"
    configs = read_config_file(config_path)
    run_single_file(configs)
