import os
import json
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import AutoMinorLocator
from math import ceil
from PIL import Image
from io import BytesIO
import pickle
from PIL import Image

standard_values = {
    "y_grid_size": 0.5,
    "x_grid_size": 0.2,
    "y_grid_inch": 5 / 25.4,
    "x_grid_inch": 5 / 25.4,
    "grid_line_width": 0.5,
    "lead_name_offset": 0.5,
    "lead_fontsize": 11,
    "x_gap": 1,
    "y_gap": 0.5,
    "display_factor": 1,
    "line_width": 0.75,
    "row_height": 8,
    "dc_offset_length": 0.2,
    "lead_length": 3,
    "V1_length": 12,
    "width": 11,
    "height": 8.5,
}

# Function to plot raw ecg signal
def ecg_plot(
    ecg,
    configs,
    sample_rate,
    columns,
    rec_file_name,
    output_dir,
    resolution,
    pad_inches,
    lead_index,
    full_mode,
    store_text_bbox,
    full_header_file,
    units="",
    papersize="",
    x_gap=standard_values["x_gap"],
    y_gap=standard_values["y_gap"],
    display_factor=standard_values["display_factor"],
    line_width=standard_values["line_width"],
    title="",
    style=None,
    row_height=standard_values["row_height"],
    show_lead_name=True,
    show_grid=False,
    show_dc_pulse=False,
    y_grid=0,
    x_grid=0,
    standard_colours=False,
    bbox=False,
    print_txt=False,
    json_dict=dict(),
    start_index=-1,
    store_configs=0,
    lead_length_in_seconds=10,
):
    # Inputs :
    # ecg - Dictionary of ecg signal with lead names as keys
    # sample_rate - Sampling rate of the ecg signal
    # lead_index - Order of lead indices to be plotted
    # columns - Number of columns to be plotted in each row
    # x_gap - gap between paper x axis border and signal plot
    # y_gap - gap between paper y axis border and signal plot
    # line_width - Width of line tracing the ecg
    # title - Title of figure
    # style - Black and white or colour
    # row_height - gap between corresponding ecg rows
    # show_lead_name - Option to show lead names or skip
    # show_dc_pulse - Option to show dc pulse
    # show_grid - Turn grid on or off

    # Initialize some params
    # secs represents how many seconds of ecg are plotted
    # leads represent number of leads in the ecg
    # rows are calculated based on corresponding number of leads and number of columns
    matplotlib.use("Agg")

    # check if the ecg dict is empty
    if ecg == {}:
        return

    secs = lead_length_in_seconds

    leads = len(lead_index)

    rows = int(ceil(leads / columns))

    if full_mode != "None":
        rows += 1
        leads += 1

    # Grid calibration
    # Each big grid corresponds to 0.2 seconds and 0.5 mV
    # To do: Select grid size in a better way
    random_line_width = random.uniform(0.5, 1.5)
    random_grid_line_width = random.uniform(0.9, 1.5)
    random_lead_fontsize = random.uniform(0.7, 2.0)
    random_lead_name_offset = random.uniform(-1.0, 2.0)
    y_grid_size = standard_values["y_grid_size"]
    x_grid_size = standard_values["x_grid_size"]
    grid_line_width = standard_values["grid_line_width"] * random_grid_line_width
    lead_name_offset = standard_values["lead_name_offset"] * random_lead_name_offset
    lead_fontsize = standard_values["lead_fontsize"] * random_lead_fontsize
    line_width = standard_values["line_width"] * random_line_width

    # Set max and min coordinates to mark grid. Offset x_max slightly (i.e by 1 column width)

    width = standard_values["width"]
    height = standard_values["height"]

    y_grid = standard_values["y_grid_inch"]
    x_grid = standard_values["x_grid_inch"]
    y_grid_dots = y_grid * resolution
    x_grid_dots = x_grid * resolution

    # row_height = height * y_grid_size/(y_grid*(rows+2))
    row_height = (height * y_grid_size / y_grid) / (rows + 2)
    x_max = width * x_grid_size / x_grid
    x_min = 0
    x_gap = np.floor(((x_max - (columns * secs)) / 2) / 0.2) * 0.2
    y_min = 0
    y_max = height * y_grid_size / y_grid

    json_dict["width"] = int(width * resolution)
    json_dict["height"] = int(height * resolution)
    # Set figure and subplot sizes
    fig, ax = plt.subplots(figsize=(width, height), dpi=resolution)

    fig.subplots_adjust(hspace=0, wspace=0, left=0, right=1, bottom=0, top=1)

    fig.suptitle(title)

    # Mark grid based on whether we want black and white or colour

    major_random_color_sampler_red = random.uniform(0.5, 0.8)
    major_random_color_sampler_green = random.uniform(0, 0.5)
    major_random_color_sampler_blue = random.uniform(0, 0.5)

    minor_offset = random.uniform(0, 0.2)
    minor_random_color_sampler_red = major_random_color_sampler_red + minor_offset
    minor_random_color_sampler_green = random.uniform(0, 0.5) + minor_offset
    minor_random_color_sampler_blue = random.uniform(0, 0.5) + minor_offset

    grey_random_color = random.uniform(0, 0.2)
    color_major = (
        major_random_color_sampler_red,
        major_random_color_sampler_green,
        major_random_color_sampler_blue,
    )
    color_minor = (
        minor_random_color_sampler_red,
        minor_random_color_sampler_green,
        minor_random_color_sampler_blue,
    )

    color_line = (grey_random_color, grey_random_color, grey_random_color)
    # color_line = (major_random_color_sampler_red,major_random_color_sampler_green,major_random_color_sampler_blue)

    # Set grid
    # Standard ecg has grid size of 0.5 mV and 0.2 seconds. Set ticks accordingly

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")

    # Step size will be number of seconds per sample i.e 1/sampling_rate
    step = 1.0 / sample_rate

    dc_offset = 0
    if show_dc_pulse:
        dc_offset = sample_rate * standard_values["dc_offset_length"] * step
    # Iterate through each lead in lead_index array.
    y_offset = row_height / 2
    x_offset = 0

    leads_ds = []

    leadNames_12 = configs["leadNames_12"]
    tickLength = configs["tickLength"]
    tickSize_step = configs["tickSize_step"]

    for i in np.arange(len(lead_index)):
        current_lead_ds = dict()

        if len(lead_index) == 12:
            leadName = leadNames_12[i]
        else:
            leadName = lead_index[i]
        # y_offset is computed by shifting by a certain offset based on i, and also by row_height/2 to account for half the waveform below the axis
        if i % columns == 0:

            y_offset += row_height

        # x_offset will be distance by which we shift the plot in each iteration
        if columns > 1:
            x_offset = (i % columns) * secs

        else:
            x_offset = 0

        # Create dc pulse wave to plot at the beginning of plot. Dc pulse will be 0.2 seconds
        x_range = np.arange(
            0, sample_rate * standard_values["dc_offset_length"] * step + 4 * step, step
        )
        dc_pulse = np.ones(len(x_range))
        dc_pulse = np.concatenate(((0, 0), dc_pulse[2:-2], (0, 0)))

        # Print lead name at .5 ( or 5 mm distance) from plot
        if show_lead_name:
            t1 = ax.text(
                x_offset + x_gap + dc_offset,
                y_offset - lead_name_offset - 0.2,
                leadName,
                fontsize=lead_fontsize,
            )

            if store_text_bbox:
                renderer1 = fig.canvas.get_renderer()
                transf = ax.transData.inverted()
                bb = t1.get_window_extent()
                x1 = bb.x0 * resolution / fig.dpi
                y1 = bb.y0 * resolution / fig.dpi
                x2 = bb.x1 * resolution / fig.dpi
                y2 = bb.y1 * resolution / fig.dpi
                box_dict = dict()
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                box_dict[0] = [round(json_dict["height"] - y2, 2), round(x1, 2)]
                box_dict[1] = [round(json_dict["height"] - y2, 2), round(x2, 2)]
                box_dict[2] = [round(json_dict["height"] - y1, 2), round(x2, 2)]
                box_dict[3] = [round(json_dict["height"] - y1, 2), round(x1, 2)]
                current_lead_ds["text_bounding_box"] = box_dict

        current_lead_ds["lead_name"] = leadName

        # If we are plotting the first row-1 plots, we plot the dc pulse prior to adding the waveform
        if columns == 1 and i in np.arange(0, rows):
            if show_dc_pulse:
                # Plot dc pulse for 0.2 seconds with 2 trailing and leading zeros to get the pulse
                t1 = ax.plot(
                    x_range + x_offset + x_gap,
                    dc_pulse + y_offset,
                    linewidth=line_width * 1.5,
                    color=color_line,
                )
                if bbox:
                    renderer1 = fig.canvas.get_renderer()
                    transf = ax.transData.inverted()
                    bb = t1[0].get_window_extent()
                    x1, y1 = bb.x0 * resolution / fig.dpi, bb.y0 * resolution / fig.dpi
                    x2, y2 = bb.x1 * resolution / fig.dpi, bb.y1 * resolution / fig.dpi

        elif i % columns == 0:
            if show_dc_pulse:
                # Plot dc pulse for 0.2 seconds with 2 trailing and leading zeros to get the pulse
                t1 = ax.plot(
                    np.arange(
                        0,
                        sample_rate * standard_values["dc_offset_length"] * step
                        + 4 * step,
                        step,
                    )
                    + x_offset
                    + x_gap,
                    dc_pulse + y_offset,
                    linewidth=line_width * 1.5,
                    color=color_line,
                )
                # print(dc_pulse,y_offset)
                if bbox:
                    renderer1 = fig.canvas.get_renderer()
                    transf = ax.transData.inverted()
                    bb = t1[0].get_window_extent()
                    x1, y1 = bb.x0 * resolution / fig.dpi, bb.y0 * resolution / fig.dpi
                    x2, y2 = bb.x1 * resolution / fig.dpi, bb.y1 * resolution / fig.dpi

        t1 = ax.plot(
            np.arange(0, len(ecg[leadName]) * step, step)
            + x_offset
            + dc_offset
            + x_gap,
            ecg[leadName] + y_offset,
            linewidth=line_width,
            color=color_line,
        )

        x_vals = (
            np.arange(0, len(ecg[leadName]) * step, step) + x_offset + dc_offset + x_gap
        )
        y_vals = ecg[leadName] + y_offset

        if bbox:
            renderer1 = fig.canvas.get_renderer()
            transf = ax.transData.inverted()
            bb = t1[0].get_window_extent()
            if show_dc_pulse == False or (
                columns == 4 and (i != 0 and i != 4 and i != 8)
            ):
                x1, y1 = bb.x0 * resolution / fig.dpi, bb.y0 * resolution / fig.dpi
                x2, y2 = bb.x1 * resolution / fig.dpi, bb.y1 * resolution / fig.dpi
            else:
                y1 = min(y1, bb.y0 * resolution / fig.dpi)
                y2 = max(y2, bb.y1 * resolution / fig.dpi)
                x2 = bb.x1 * resolution / fig.dpi
            box_dict = dict()
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            box_dict[0] = [round(json_dict["height"] - y2, 2), round(x1, 2)]
            box_dict[1] = [round(json_dict["height"] - y2, 2), round(x2, 2)]
            box_dict[2] = [round(json_dict["height"] - y1, 2), round(x2, 2)]
            box_dict[3] = [round(json_dict["height"] - y1, 2), round(x1, 2)]
            current_lead_ds["lead_bounding_box"] = box_dict

        st = start_index
        if columns == 4 and leadName in configs["format_4_by_3"][1]:
            st = start_index + int(sample_rate * configs["paper_len"] / columns)
        elif columns == 4 and leadName in configs["format_4_by_3"][2]:
            st = start_index + int(2 * sample_rate * configs["paper_len"] / columns)
        elif columns == 4 and leadName in configs["format_4_by_3"][3]:
            st = start_index + int(3 * sample_rate * configs["paper_len"] / columns)
        current_lead_ds["start_sample"] = st
        current_lead_ds["end_sample"] = st + len(ecg[leadName])
        current_lead_ds["plotted_pixels"] = []
        for j in range(len(x_vals)):
            xi, yi = x_vals[j], y_vals[j]
            xi, yi = ax.transData.transform((xi, yi))
            yi = json_dict["height"] - yi
            current_lead_ds["plotted_pixels"].append([round(yi, 2), round(xi, 2)])
        leads_ds.append(current_lead_ds)
        if random.random() < 0.5:
            if columns > 1 and (i + 1) % columns != 0:
                # sep_x = [len(ecg[leadName])*step + x_offset + dc_offset + x_gap] * round(8*y_grid_dots)
                sep_x = [
                    len(ecg[leadName]) * step + x_offset + dc_offset + x_gap
                ] * round(tickLength * y_grid_dots)
                sep_x = np.array(sep_x)
                # sep_y = np.linspace(y_offset - 4*y_grid_dots*step, y_offset + 4*y_grid_dots*step, len(sep_x))
                sep_y = np.linspace(
                    y_offset - tickLength / 2 * y_grid_dots * tickSize_step,
                    y_offset + tickSize_step * y_grid_dots * tickLength / 2,
                    len(sep_x),
                )
                ax.plot(sep_x, sep_y, linewidth=line_width * 3, color=color_line)

    # Plotting longest lead for 12 seconds
    if full_mode != "None":
        current_lead_ds = dict()
        if show_lead_name:
            t1 = ax.text(
                x_gap + dc_offset,
                row_height / 2 - lead_name_offset,
                full_mode,
                fontsize=lead_fontsize,
            )

            if store_text_bbox:
                renderer1 = fig.canvas.get_renderer()
                transf = ax.transData.inverted()
                bb = t1.get_window_extent(renderer=fig.canvas.renderer)
                x1 = bb.x0 * resolution / fig.dpi
                y1 = bb.y0 * resolution / fig.dpi
                x2 = bb.x1 * resolution / fig.dpi
                y2 = bb.y1 * resolution / fig.dpi
                box_dict = dict()
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                box_dict[0] = [round(json_dict["height"] - y2, 2), round(x1, 2)]
                box_dict[1] = [round(json_dict["height"] - y2, 2), round(x2, 2)]
                box_dict[2] = [round(json_dict["height"] - y1, 2), round(x2, 2)]
                box_dict[3] = [round(json_dict["height"] - y1), round(x1, 2)]
                current_lead_ds["text_bounding_box"] = box_dict
            current_lead_ds["lead_name"] = full_mode

        if show_dc_pulse:
            t1 = ax.plot(
                x_range + x_gap,
                dc_pulse + row_height / 2 - lead_name_offset + 0.8,
                linewidth=line_width * 1.5,
                color=color_line,
            )

            # print(dc_pulse,row_height/2-lead_name_offset + 0.8)

            if bbox:
                renderer1 = fig.canvas.get_renderer()
                transf = ax.transData.inverted()
                bb = t1[0].get_window_extent()
                x1, y1 = bb.x0 * resolution / fig.dpi, bb.y0 * resolution / fig.dpi
                x2, y2 = bb.x1 * resolution / fig.dpi, bb.y1 * resolution / fig.dpi

        dc_full_lead_offset = 0
        if show_dc_pulse:
            dc_full_lead_offset = (
                sample_rate * standard_values["dc_offset_length"] * step
            )

        t1 = ax.plot(
            np.arange(0, len(ecg["full" + full_mode]) * step, step)
            + x_gap
            + dc_full_lead_offset,
            ecg["full" + full_mode] + row_height / 2 - lead_name_offset + 0.8,
            linewidth=line_width,
            color=color_line,
        )
        x_vals = (
            np.arange(0, len(ecg["full" + full_mode]) * step, step)
            + x_gap
            + dc_full_lead_offset
        )
        y_vals = ecg["full" + full_mode] + row_height / 2 - lead_name_offset + 0.8

        if bbox:
            renderer1 = fig.canvas.get_renderer()
            transf = ax.transData.inverted()
            bb = t1[0].get_window_extent()
            if show_dc_pulse == False:
                x1, y1 = bb.x0 * resolution / fig.dpi, bb.y0 * resolution / fig.dpi
                x2, y2 = bb.x1 * resolution / fig.dpi, bb.y1 * resolution / fig.dpi
            else:
                y1 = min(y1, bb.y0 * resolution / fig.dpi)
                y2 = max(y2, bb.y1 * resolution / fig.dpi)
                x2 = bb.x1 * resolution / fig.dpi

            box_dict = dict()
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            box_dict[0] = [round(json_dict["height"] - y2, 2), round(x1, 2)]
            box_dict[1] = [round(json_dict["height"] - y2), round(x2, 2)]
            box_dict[2] = [round(json_dict["height"] - y1, 2), round(x2, 2)]
            box_dict[3] = [round(json_dict["height"] - y1, 2), round(x1, 2)]
            current_lead_ds["lead_bounding_box"] = box_dict
        current_lead_ds["start_sample"] = start_index
        current_lead_ds["end_sample"] = start_index + len(ecg["full" + full_mode])
        current_lead_ds["plotted_pixels"] = []
        current_lead_ds["dc_offset"] = dc_full_lead_offset
        for i in range(len(x_vals)):
            xi, yi = x_vals[i], y_vals[i]
            xi, yi = ax.transData.transform((xi, yi))
            yi = json_dict["height"] - yi
            current_lead_ds["plotted_pixels"].append([round(yi, 2), round(xi, 2)])
        leads_ds.append(current_lead_ds)

    head, tail = os.path.split(rec_file_name)
    rec_file_name = os.path.join(output_dir, tail)

    # change x and y res
    ax.text(2, 0.5, "25mm/s", fontsize=lead_fontsize)
    ax.text(4, 0.5, "10mm/mV", fontsize=lead_fontsize)

    print(color_minor, color_major)
    if np.random.rand() < 0.2:
        color_minor = (0.6, 0.6, 0.6)
        color_major = (0.3, 0.3, 0.3)

    if show_grid:
        ax.set_xticks(np.arange(x_min, x_max, x_grid_size))
        ax.set_yticks(np.arange(y_min, y_max, y_grid_size))
        ax.minorticks_on()

        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        # set grid line style
        ax.grid(
            which="major", linestyle="-", linewidth=grid_line_width, color=color_major
        )
        ax.grid(
            which="minor", linestyle="-", linewidth=grid_line_width, color=color_minor
        )

        if store_configs == 2:
            json_dict["grid_line_color_major"] = [
                round(x * 255.0, 2) for x in color_major
            ]
            json_dict["grid_line_color_minor"] = [
                round(x * 255.0, 2) for x in color_minor
            ]
            json_dict["ecg_plot_color"] = [round(x * 255.0, 2) for x in color_line]
    else:
        ax.grid(False)

    # find all dc pulses
    dc_pulses_xy_data = []  # We expect four pulses
    straight_lines_data = []
    for obj in fig.findobj(match=lambda obj: isinstance(obj, plt.Line2D)):
        xdata = obj.get_xdata()
        ydata = obj.get_ydata()
        if len(xdata) == 24:
            dc_pulse = []
            for xi, yi in zip(xdata, ydata):
                xi, yi = ax.transData.transform((xi, yi))
                yi = json_dict["height"] - yi
                dc_pulse.append([int(yi - 0.5), int(xi - 0.5)])
            dc_pulses_xy_data.append(dc_pulse)
        else:
            if len(xdata) == 315:
                straight_line = []
                for xi, yi in zip(xdata, ydata):
                    xi, yi = ax.transData.transform((xi, yi))
                    yi = json_dict["height"] - yi
                    straight_line.append([int(yi - 0.5), int(xi - 0.5)])
                straight_lines_data.append(straight_line)

    plt.savefig(os.path.join(output_dir, tail + ".png"), dpi=resolution)
    plt.close(fig)
    plt.clf()
    plt.cla()

    #######################
    def copy_figure(fig):
        """Copy a Matplotlib figure."""
        buf = BytesIO()
        pickle.dump(fig, buf)
        buf.seek(0)
        return pickle.load(buf)

    def fig_to_array(fig):
        """Convert a Matplotlib figure to a numpy array."""
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img = Image.open(buf)
        return np.array(img)

    def remove_text_objects(fig):
        """Remove text objects from a Matplotlib figure."""
        text_objects = fig.findobj(match=plt.Text)
        for text_obj in text_objects:
            text_obj.set_visible(False)
        return text_objects

    def remove_line2d_objects_315(fig):
        """Remove Line2D objects of length 315 from a Matplotlib figure."""
        line2d_objects = []
        for line in fig.findobj(match=plt.Line2D):
            if len(line.get_xydata()) == 315:
                line.set_visible(False)
                line2d_objects.append(line)
        return line2d_objects

    def remove_line2d_objects_250_1000(fig):
        """Remove Line2D objects of length 315 from a Matplotlib figure."""
        line2d_objects = []
        for line in fig.findobj(match=plt.Line2D):
            if len(line.get_xydata()) in (250, 1000):
                line.set_visible(False)
                line2d_objects.append(line)
        return line2d_objects

    def remove_line2d_objects_24(fig):
        """Remove Line2D objects of length 315 from a Matplotlib figure."""
        line2d_objects = []
        for line in fig.findobj(match=plt.Line2D):
            if len(line.get_xydata()) == 24:
                line.set_visible(False)
                line2d_objects.append(line)
        return line2d_objects

    def restore_objects(objects):
        """Restore visibility of given objects in a Matplotlib figure."""
        for obj in objects:
            obj.set_visible(True)

    def render_text_layer(fig):
        """Render only the text from a Matplotlib figure on a blank canvas."""

        # Step 1: Copy the figure
        fig_copy = copy_figure(fig)
        # Step 2: Render the original figure to a numpy array
        original_array = fig_to_array(fig_copy)
        # Step 3: Remove everything plotted
        removed_texts = remove_text_objects(fig_copy)
        removed_lines_315 = remove_line2d_objects_315(fig_copy)
        removed_control_signal_lines = remove_line2d_objects_24(fig_copy)
        removed_signal_lines = remove_line2d_objects_250_1000(fig_copy)
        for line in fig_copy.findobj(match=plt.Line2D):
            if line.get_linewidth() == grid_line_width:
                line.set_visible(False)

        # Step 4: Render figure with only specified parts
        restore_objects(removed_texts)
        restore_objects(removed_lines_315)
        greyscale_text = 255 - np.max(fig_to_array(fig_copy)[..., :3], axis=-1)
        removed_texts = remove_text_objects(fig_copy)
        removed_lines_315 = remove_line2d_objects_315(fig_copy)

        restore_objects(removed_control_signal_lines)
        greyscale_control_signal = 255 - np.max(
            fig_to_array(fig_copy)[..., :3], axis=-1
        )
        removed_control_signal_lines = remove_line2d_objects_24(fig_copy)

        restore_objects(removed_signal_lines)
        greyscale_signal = 255 - np.max(fig_to_array(fig_copy)[..., :3], axis=-1)
        removed_signal_lines = remove_line2d_objects_250_1000(fig_copy)

        # Step 6: Restore the visibility of the removed objects
        for line in fig_copy.findobj(match=plt.Line2D):
            if line.get_linewidth() == grid_line_width:
                line.set_visible(True)
        restore_objects(removed_texts)
        restore_objects(removed_lines_315)
        restore_objects(removed_control_signal_lines)
        restore_objects(removed_signal_lines)

        # fig, ax = plt.subplots(1,3)
        # ax[0].imshow(greyscale_signal, cmap='gray')
        # ax[1].imshow(greyscale_control_signal, cmap='gray')
        # ax[2].imshow(greyscale_text, cmap='gray')
        # for a in ax:
        #     a.axis('off')
        # plt.tight_layout()

        # plt.savefig(os.path.join(output_dir, tail + '_greyscale.png'), dpi=800)
        # assert False

        return greyscale_signal, greyscale_control_signal, greyscale_text

    #######################
    signal_map, control_signal_map, text_map = render_text_layer(fig)

    if pad_inches != 0:

        ecg_image = Image.open(os.path.join(output_dir, tail + ".png"))

        right = pad_inches * resolution
        left = pad_inches * resolution
        top = pad_inches * resolution
        bottom = pad_inches * resolution
        width, height = ecg_image.size
        new_width = width + right + left
        new_height = height + top + bottom
        result_image = Image.new(
            ecg_image.mode, (new_width, new_height), (255, 255, 255)
        )
        result_image.paste(ecg_image, (left, top))

        result_image.save(os.path.join(output_dir, tail + ".png"))

        plt.close("all")
        plt.close(fig)
        plt.clf()
        plt.cla()

    json_dict["leads"] = leads_ds

    ##### MODIFICATIONS #####
    json_file = os.path.join(output_dir, tail + ".json")
    with open(json_file, "w") as f:
        json.dump(json_dict, f)

    w, h = json_dict["width"], json_dict["height"]
    mask = np.zeros((h, w, 3), dtype=np.uint8)  # Initialize as uint8 for 0 or 1 values

    mask[:, :, 2] = signal_map.astype(np.uint8)
    mask[:, :, 1] = text_map.astype(np.uint8) + control_signal_map.astype(np.uint8)
    mask[:, :3] = 0
    mask[:, -3:] = 0
    mask[:3, :] = 0
    mask[-3:, :] = 0
    mask[:, :, 1][mask[:, :, 2] > 0] = 0

    # Save the sparse tensor
    mask_file = os.path.join(output_dir, tail + "mask_.png")
    # save the mask as a png at mask_file
    print(mask.shape, mask.dtype)
    Image.fromarray(mask).save(mask_file)
    ###### END MODIFICATIONS ######

    return x_grid_dots, y_grid_dots
