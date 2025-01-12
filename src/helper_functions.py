import os
import yaml
import numpy as np
import wfdb
from scipy.io import loadmat

BIT_NAN_16 = -(2.0**15)


def read_config_file(config_file):
    """Read YAML config file

    Args:
        config_file (str): Complete path to the config file

    Returns:
        configs (dict): Returns dictionary with all the configs
    """
    with open(config_file) as f:
        yamlObject = yaml.safe_load(f)

    args = dict()
    for key in yamlObject:
        args[key] = yamlObject[key]

    return args


def load_header(header_file):
    with open(header_file, "r") as f:
        header = f.read()
    return header


def load_recording(recording_file, header=None, key="val"):
    rootname, extension = os.path.splitext(recording_file)
    if extension == ".dat":
        recording = wfdb.rdrecord(rootname)
        return recording.p_signal
    if extension == ".mat":
        recording = loadmat(recording_file)[key]
    return recording


def get_leads(header):
    leads = list()
    for i, l in enumerate(header.split("\n")):
        entries = l.split(" ")
        if i == 0:
            num_leads = int(entries[1])
        elif i <= num_leads:
            leads.append(entries[-1])
        else:
            break
    return tuple(leads)


def get_frequency(header):
    frequency = None
    for i, l in enumerate(header.split("\n")):
        if i == 0:
            try:
                frequency = l.split(" ")[2]
                if "/" in frequency:
                    frequency = float(frequency.split("/")[0])
                else:
                    frequency = float(frequency)
            except:
                pass
        else:
            break
    return frequency


def create_signal_dictionary(signal, full_leads):
    record_dict = {}
    for k in range(len(full_leads)):
        record_dict[full_leads[k]] = signal[k]

    return record_dict


def standardize_leads(full_leads):
    full_leads_array = np.asarray(full_leads)

    for i in np.arange(len(full_leads_array)):
        if full_leads_array[i].upper() not in ("AVR", "AVL", "AVF"):
            full_leads_array[i] = full_leads_array[i].upper()
        else:
            if full_leads_array[i].upper() == "AVR":
                full_leads_array[i] = "aVR"
            elif full_leads_array[i].upper() == "AVL":
                full_leads_array[i] = "aVL"
            else:
                full_leads_array[i] = "aVF"
    return full_leads_array


def read_leads(leads):

    lead_bbs = []
    text_bbs = []
    startTimeStamps = []
    endTimeStamps = []
    labels = []
    plotted_pixels = []
    for i, line in enumerate(leads):
        labels.append(leads[i]["lead_name"])
        st_time_stamp = leads[i]["start_sample"]
        startTimeStamps.append(st_time_stamp)
        end_time_stamp = leads[i]["end_sample"]
        endTimeStamps.append(end_time_stamp)
        plotted_pixels.append(leads[i]["plotted_pixels"])

        key = "lead_bounding_box"
        if key in leads[i].keys():
            parts = leads[i][key]
            point1 = [parts["0"][0], parts["0"][1]]
            point2 = [parts["1"][0], parts["1"][1]]
            point3 = [parts["2"][0], parts["2"][1]]
            point4 = [parts["3"][0], parts["3"][1]]
            box = [point1, point2, point3, point4]
            lead_bbs.append(box)

        key = "text_bounding_box"
        if key in leads[i].keys():
            parts = leads[i][key]
            point1 = [parts["0"][0], parts["0"][1]]
            point2 = [parts["1"][0], parts["1"][1]]
            point3 = [parts["2"][0], parts["2"][1]]
            point4 = [parts["3"][0], parts["3"][1]]
            box = [point1, point2, point3, point4]
            text_bbs.append(box)

    if len(lead_bbs) != 0:
        lead_bbs = np.array(lead_bbs)
    if len(text_bbs) != 0:
        text_bbs = np.array(text_bbs)

    return (
        lead_bbs,
        text_bbs,
        labels,
        startTimeStamps,
        endTimeStamps,
        plotted_pixels,
    )


def write_wfdb_file(
    ecg_frame,
    filename,
    rate,
    header_file,
    write_dir,
    full_mode,
    mask_unplotted_samples,
):
    full_header = load_header(header_file)
    full_leads = get_leads(full_header)
    full_leads = standardize_leads(full_leads)

    samples = len(ecg_frame[full_mode])
    array = np.zeros((1, samples))

    leads = []
    header_name, extn = os.path.splitext(header_file)
    header = wfdb.rdheader(header_name)

    for i, lead in enumerate(full_leads):
        leads.append(lead)
        if lead == full_mode:
            lead = "full" + lead
        adc_gn = header.adc_gain[i]

        arr = ecg_frame[lead]
        arr = np.array(arr)
        arr[np.isnan(arr)] = BIT_NAN_16 / adc_gn
        arr = arr.reshape((1, arr.shape[0]))
        array = np.concatenate((array, arr), axis=0)

    head, tail = os.path.split(filename)

    array = array[1:]
    wfdb.wrsamp(
        record_name=tail,
        fs=rate,
        units=header.units,
        sig_name=leads,
        p_signal=array.T,
        fmt=header.fmt,
        adc_gain=header.adc_gain,
        baseline=header.baseline,
        base_time=header.base_time,
        base_date=header.base_date,
        write_dir=write_dir,
        comments=header.comments,
    )
