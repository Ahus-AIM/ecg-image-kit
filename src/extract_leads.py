#!/usr/bin/env python
# Load libraries.
import os
import json
import numpy as np
from helper_functions import (
    get_frequency,
    get_leads,
    load_recording,
    load_header,
    create_signal_dictionary,
    standardize_leads,
    write_wfdb_file,
)
from ecg_plot import ecg_plot
from scipy.signal import butter, lfilter


def bandpass_filter(signal, rate, lowcut, highcut, pad_len=2048):
    """
    Bandpass filter the data between lowcut and highcut.
    """
    padded_signal = np.pad(signal, pad_len, mode="reflect")
    nyquist = 0.5 * rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(1, [low, high], btype="band")
    padded_filtered = lfilter(b, a, padded_signal, axis=0)
    return padded_filtered[pad_len:-pad_len]


def get_paper_ecg(
    input_file,
    header_file,
    output_directory,
    seed,
    add_dc_pulse,
    add_bw,
    show_grid,
    add_print,
    configs,
    mask_unplotted_samples=False,
    start_index=-1,
    store_configs=False,
    store_text_bbox=True,
    key="val",
    resolution=100,
    units="inches",
    papersize="",
    add_lead_names=True,
    pad_inches=1,
    template_file=os.path.join("TemplateFiles", "TextFile1.txt"),
    font_type=os.path.join("Fonts", "Times_New_Roman.ttf"),
    standard_colours=5,
    full_mode="II",
    bbox=False,
    columns=-1,
):

    # Extract a reduced-lead set from each pair of full-lead header and recording files.
    full_header_file = header_file
    full_recording_file = input_file
    full_header = load_header(full_header_file)
    full_leads = get_leads(full_header)
    num_full_leads = len(full_leads)

    # Update the header file
    full_lines = full_header.split("\n")

    head, tail = os.path.split(full_header_file)

    # Load the full-lead recording file, extract the lead data, and save the reduced-lead recording file.
    recording = load_recording(full_recording_file, full_header, key)

    # Get values from header
    rate = get_frequency(full_header)

    full_leads = standardize_leads(full_leads)

    recording = bandpass_filter(recording, rate, 0.5, min(150, rate / 2 - 1))
    if recording.max() > 10 or recording.min() < -10:
        recording = recording * 10 / max(abs(recording.max()), abs(recording.min()))
    if recording.shape[0] < rate * 10:
        newline = "\n"
        print(
            f"Recording is too short to plot (less than 10 seconds), skipping. First row of header file: {full_header.split(newline)[0]}"
        )
        return None
    if rate not in [100.0, 400.0]:
        newline = "\n"
        print(
            f"Sampling frequency is not 100 or 400 Hz, skipping. First row of header file: {full_header.split(newline)[0]}"
        )
        return None
    output_header_file = os.path.join(output_directory, tail)
    with open(output_header_file, "w") as f:
        f.write("\n".join(full_lines))

    if len(full_leads) == 2:
        full_mode = "None"
        if columns == -1:
            columns = 1

    elif len(full_leads) == 12:
        if full_mode not in full_leads:
            full_mode = full_leads[0]
        else:
            full_mode = full_mode
        if columns == -1:
            columns = 4
    else:
        columns = 4
        full_mode = "None"

    if recording.shape[0] != num_full_leads:
        recording = np.transpose(recording)

    record_dict = create_signal_dictionary(recording, full_leads)

    gain_index = 0

    ecg_frame = []
    end_flag = False
    start = 0
    lead_length_in_seconds = 10 / columns
    abs_lead_step = 10
    format_4_by_3 = [
        ["I", "II", "III"],
        ["aVR", "aVL", "aVF", "AVR", "AVL", "AVF"],
        ["V1", "V2", "V3"],
        ["V4", "V5", "V6"],
    ]

    segmented_ecg_data = {}

    if start_index != -1:
        start = start_index
        # do something
        frame = {}
        gain_index = 0
        for key in record_dict:
            if len(record_dict[key][start:]) < int(rate * abs_lead_step):
                end_flag = True
                nanArray = np.empty(len(record_dict[key][start:]))
                if mask_unplotted_samples:
                    nanArray[:] = np.nan
                else:
                    nanArray[:] = record_dict[key][start:]
                if full_mode != "None" and key == full_mode:
                    if "full" + full_mode not in segmented_ecg_data.keys():
                        segmented_ecg_data["full" + full_mode] = nanArray.tolist()
                    else:
                        segmented_ecg_data["full" + full_mode] = (
                            segmented_ecg_data["full" + full_mode] + nanArray.tolist()
                        )
                if key != "full" + full_mode:
                    if key not in segmented_ecg_data.keys():
                        segmented_ecg_data[key] = nanArray.tolist()
                    else:
                        segmented_ecg_data[key] = (
                            segmented_ecg_data[key] + nanArray.tolist()
                        )
            else:
                shiftedStart = start
                if columns == 4 and key in format_4_by_3[1]:
                    shiftedStart = start + int(rate * lead_length_in_seconds)
                elif columns == 4 and key in format_4_by_3[2]:
                    shiftedStart = start + int(2 * rate * lead_length_in_seconds)
                elif columns == 4 and key in format_4_by_3[3]:
                    shiftedStart = start + int(3 * rate * lead_length_in_seconds)
                end = shiftedStart + int(rate * lead_length_in_seconds)

                if key != "full" + full_mode:
                    frame[key] = record_dict[key][shiftedStart:end]
                    frame[key] = frame[key]

                    nanArray = np.empty((int(shiftedStart - start)))
                    if mask_unplotted_samples:
                        nanArray[:] = np.nan
                    else:
                        nanArray[:] = record_dict[key][start:shiftedStart]
                    if columns == 4 and key not in format_4_by_3[0]:
                        if key not in segmented_ecg_data.keys():
                            segmented_ecg_data[key] = nanArray.tolist()
                        else:
                            segmented_ecg_data[key] = (
                                segmented_ecg_data[key] + nanArray.tolist()
                            )
                    if key not in segmented_ecg_data.keys():
                        segmented_ecg_data[key] = frame[key].tolist()
                    else:
                        segmented_ecg_data[key] = (
                            segmented_ecg_data[key] + frame[key].tolist()
                        )

                    nanArray = np.empty(
                        (
                            int(
                                abs_lead_step * rate
                                - (end - shiftedStart)
                                - (shiftedStart - start)
                            )
                        )
                    )
                    nanArray_len = int(
                        abs_lead_step * rate
                        - (end - shiftedStart)
                        - (shiftedStart - start)
                    )
                    if mask_unplotted_samples:
                        nanArray[:] = np.nan
                    else:
                        nanArray[:] = record_dict[key][end : end + nanArray_len]
                    segmented_ecg_data[key] = (
                        segmented_ecg_data[key] + nanArray.tolist()
                    )
                if full_mode != "None" and key == full_mode:
                    if len(record_dict[key][start:]) > int(rate * 10):
                        frame["full" + full_mode] = record_dict[key][
                            start : (start + int(rate) * 10)
                        ]
                        frame["full" + full_mode] = frame["full" + full_mode]
                        if "full" + full_mode not in segmented_ecg_data.keys():
                            segmented_ecg_data["full" + full_mode] = frame[
                                "full" + full_mode
                            ].tolist()
                        else:
                            segmented_ecg_data["full" + full_mode] = (
                                segmented_ecg_data["full" + full_mode]
                                + frame["full" + full_mode].tolist()
                            )
                    else:
                        frame["full" + full_mode] = record_dict[key][start:]
                        frame["full" + full_mode] = frame["full" + full_mode]
                        if "full" + full_mode not in segmented_ecg_data.keys():
                            segmented_ecg_data["full" + full_mode] = frame[
                                "full" + full_mode
                            ].tolist()
                        else:
                            segmented_ecg_data["full" + full_mode] = (
                                segmented_ecg_data["full" + full_mode]
                                + frame["full" + full_mode].tolist()
                            )
                gain_index += 1
        ecg_frame.append(frame)

    else:
        while not end_flag:
            # To do : Incorporate column and ful_mode info
            frame = {}
            gain_index = 0

            for key in record_dict:
                if len(record_dict[key][start:]) < int(rate * abs_lead_step):
                    end_flag = True
                    nanArray = np.empty(len(record_dict[key][start:]))
                    if mask_unplotted_samples:
                        nanArray[:] = np.nan
                    else:
                        nanArray[:] = record_dict[key][start:]

                    if full_mode != "None" and key == full_mode:
                        if "full" + full_mode not in segmented_ecg_data.keys():
                            segmented_ecg_data["full" + full_mode] = nanArray.tolist()
                        else:
                            segmented_ecg_data["full" + full_mode] = (
                                segmented_ecg_data["full" + full_mode]
                                + nanArray.tolist()
                            )
                    if key != "full" + full_mode:
                        if key not in segmented_ecg_data.keys():
                            segmented_ecg_data[key] = nanArray.tolist()
                        else:
                            segmented_ecg_data[key] = (
                                segmented_ecg_data[key] + nanArray.tolist()
                            )
                else:
                    shiftedStart = start
                    if columns == 4 and key in format_4_by_3[1]:
                        shiftedStart = start + int(rate * lead_length_in_seconds)
                    elif columns == 4 and key in format_4_by_3[2]:
                        shiftedStart = start + int(2 * rate * lead_length_in_seconds)
                    elif columns == 4 and key in format_4_by_3[3]:
                        shiftedStart = start + int(3 * rate * lead_length_in_seconds)
                    end = shiftedStart + int(rate * lead_length_in_seconds)

                    if key != "full" + full_mode:
                        frame[key] = record_dict[key][shiftedStart:end]
                        frame[key] = frame[key]

                        nanArray = np.empty((int(shiftedStart - start)))
                        if mask_unplotted_samples:
                            nanArray[:] = np.nan
                        else:
                            nanArray[:] = record_dict[key][start:shiftedStart]

                        if columns == 4 and key not in format_4_by_3[0]:
                            if key not in segmented_ecg_data.keys():
                                segmented_ecg_data[key] = nanArray.tolist()
                            else:
                                segmented_ecg_data[key] = (
                                    segmented_ecg_data[key] + nanArray.tolist()
                                )
                        if key not in segmented_ecg_data.keys():
                            segmented_ecg_data[key] = frame[key].tolist()
                        else:
                            segmented_ecg_data[key] = (
                                segmented_ecg_data[key] + frame[key].tolist()
                            )

                        nanArray = np.empty(
                            (
                                int(
                                    abs_lead_step * rate
                                    - (end - shiftedStart)
                                    - (shiftedStart - start)
                                )
                            )
                        )
                        nanArray_len = int(
                            abs_lead_step * rate
                            - (end - shiftedStart)
                            - (shiftedStart - start)
                        )
                        if mask_unplotted_samples:
                            nanArray[:] = np.nan
                        else:
                            nanArray[:] = record_dict[key][end : end + nanArray_len]

                        segmented_ecg_data[key] = (
                            segmented_ecg_data[key] + nanArray.tolist()
                        )
                    if full_mode != "None" and key == full_mode:
                        if len(record_dict[key][start:]) > int(rate * 10):
                            frame["full" + full_mode] = record_dict[key][
                                start : (start + int(rate) * 10)
                            ]
                            frame["full" + full_mode] = frame["full" + full_mode]
                            if "full" + full_mode not in segmented_ecg_data.keys():
                                segmented_ecg_data["full" + full_mode] = frame[
                                    "full" + full_mode
                                ].tolist()
                            else:
                                segmented_ecg_data["full" + full_mode] = (
                                    segmented_ecg_data["full" + full_mode]
                                    + frame["full" + full_mode].tolist()
                                )
                        else:
                            frame["full" + full_mode] = record_dict[key][start:]
                            frame["full" + full_mode] = frame["full" + full_mode]
                            if "full" + full_mode not in segmented_ecg_data.keys():
                                segmented_ecg_data["full" + full_mode] = frame[
                                    "full" + full_mode
                                ].tolist()
                            else:
                                segmented_ecg_data["full" + full_mode] = (
                                    segmented_ecg_data["full" + full_mode]
                                    + frame["full" + full_mode].tolist()
                                )
                    gain_index += 1
            if not end_flag:
                ecg_frame.append(frame)
                start = start + int(rate * abs_lead_step)
    outfile_array = []

    name, ext = os.path.splitext(full_header_file)
    write_wfdb_file(
        segmented_ecg_data,
        name,
        rate,
        header_file,
        output_directory,
        full_mode,
        mask_unplotted_samples,
    )

    if len(ecg_frame) == 0:
        return outfile_array

    start = 0
    for i in range(len(ecg_frame)):
        dc = add_dc_pulse.rvs()
        bw = add_bw.rvs()
        grid = show_grid.rvs()
        print_txt = add_print.rvs()

        json_dict = {}
        json_dict["sampling_frequency"] = rate
        grid_colour = "colour"
        if bw:
            grid_colour = "bw"

        rec_file = name + "-" + str(i)
        if ecg_frame[i] == {}:
            continue

        x_grid, y_grid = ecg_plot(
            ecg_frame[i],
            configs=configs,
            full_header_file=full_header_file,
            style=grid_colour,
            sample_rate=rate,
            columns=columns,
            rec_file_name=rec_file,
            output_dir=output_directory,
            resolution=resolution,
            pad_inches=pad_inches,
            lead_index=full_leads,
            full_mode=full_mode,
            store_text_bbox=store_text_bbox,
            show_lead_name=add_lead_names,
            show_dc_pulse=dc,
            papersize=papersize,
            show_grid=(grid),
            standard_colours=standard_colours,
            bbox=bbox,
            print_txt=print_txt,
            json_dict=json_dict,
            start_index=start,
            store_configs=store_configs,
            lead_length_in_seconds=lead_length_in_seconds,
        )

        rec_head, rec_tail = os.path.split(rec_file)

        json_dict["x_grid"] = round(x_grid, 3)
        json_dict["y_grid"] = round(y_grid, 3)
        json_dict["resolution"] = resolution
        json_dict["pad_inches"] = pad_inches

        if store_configs == 2:
            json_dict["dc_pulse"] = bool(dc)
            json_dict["bw"] = bool(bw)
            json_dict["gridlines"] = bool(grid)
            json_dict["printed_text"] = bool(print_txt)
            json_dict["number_of_columns_in_image"] = columns
            json_dict["full_mode_lead"] = full_mode

        outfile = os.path.join(output_directory, rec_tail + ".png")

        json_object = json.dumps(json_dict, indent=4)

        # Writing to sample.json
        if store_configs:
            with open(os.path.join(output_directory, rec_tail + ".json"), "w") as f:
                f.write(json_object)

        outfile_array.append(outfile)
        start += int(rate * abs_lead_step)
    return outfile_array
