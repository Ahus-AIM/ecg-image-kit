import os
import sys
import argparse
import random
import csv
import warnings
from scipy.stats import bernoulli
from extract_leads import get_paper_ecg
from CreasesWrinkles.creases import get_creased
from helper_functions import read_config_file

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, required=True)
    parser.add_argument("-hea", "--header_file", type=str, required=True)
    parser.add_argument("-o", "--output_directory", type=str, required=True)
    parser.add_argument("--config_file", type=str, default="config.yaml")
    parser.add_argument("--wrinkles", action="store_true", default=False)

    return parser



def run_single_file(args):
    args.encoding = args.input_file

    filename = args.input_file
    header = args.header_file
    resolution = 200
    padding=0

    papersize = ""
    lead = True

    bernoulli_dc = bernoulli(0.5)
    bernoulli_bw = bernoulli(0)
    bernoulli_grid = bernoulli(1)
    bernoulli_add_print = bernoulli(0)

    font = os.path.join("Fonts", random.choice(os.listdir("Fonts")))

    standard_colours = False

    configs = read_config_file(os.path.join(os.getcwd(), args.config_file))

    out_array = get_paper_ecg(
        input_file=filename,
        header_file=header,
        configs=configs,
        mask_unplotted_samples=False,
        start_index=-1,
        store_configs=0,
        store_text_bbox=False,
        output_directory=args.output_directory,
        resolution=resolution,
        papersize=papersize,
        add_lead_names=lead,
        add_dc_pulse=bernoulli_dc,
        add_bw=bernoulli_bw,
        show_grid=bernoulli_grid,
        add_print=bernoulli_add_print,
        pad_inches=0,
        font_type=font,
        standard_colours=standard_colours,
        full_mode='II',
        bbox=False,
        columns=-1,
        seed=42,
    )

    for out in out_array:
        rec_tail, extn = os.path.splitext(out)
        wrinkles = args.wrinkles

        if wrinkles:
            ifWrinkles = True
            ifCreases = True
            crease_angle = random.choice(range(0, 90))
            num_creases_vertically = random.choice(range(1, 5))
            num_creases_horizontally = random.choice(range(1, 5))
            out = get_creased(
                out,
                output_directory=args.output_directory,
                ifWrinkles=ifWrinkles,
                ifCreases=ifCreases,
                crease_angle=crease_angle,
                num_creases_vertically=num_creases_vertically,
                num_creases_horizontally=num_creases_horizontally,
                bbox=False,
            )
        else:
            crease_angle = 0
            num_creases_horizontally = 0
            num_creases_vertically = 0

    return len(out_array)


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), sys.argv[0])
    parentPath = os.path.dirname(path)
    os.chdir(parentPath)
    run_single_file(get_parser().parse_args(sys.argv[1:]))
