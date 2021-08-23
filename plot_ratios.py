#!/usr/bin/python3

from zipfile import ZipFile
import json
import sys
import argparse
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import poisson
from scipy.optimize import curve_fit


def get_counts(zip, filenames):
    dicts = [json.loads(zip.read(fn)) for fn in filenames]

    message_counts = defaultdict(lambda: 0)
    for d in dicts:
        for m in d["messages"]:
            name = m["sender_name"]
            if name.startswith("Other ") or name.startswith("Facebook"):
                continue
            message_counts[name] += 1

    return message_counts


def get_counts_from_zip(zip_name):
    counts = []
    with ZipFile(zip_name, "r") as zip:
        conv_files = []
        for f in zip.filelist:
            filename = f.filename
            if filename.startswith("messages/inbox") and filename.endswith(".json"):
                conv_files.append(filename)

        conv_file_locs = set("/".join(filename.split("/")[:-1]) for filename in conv_files)

        for loc in set(conv_file_locs):
            c = get_counts(zip, [filename for filename in conv_files if filename.startswith(loc)])
            counts.append(c)
    return counts


def get_ratios_from_counts(counts, user_name, min_msg_count=20, min_participant_count=2):
    ratios = []
    for message_counts in counts:
        message_total_count = sum(v for v in message_counts.values())
        if message_total_count < min_msg_count:
            continue

        participant_count = len(message_counts)
        if participant_count < min_participant_count:
            continue

        if user_name not in message_counts:
            continue

        user_count = message_counts[user_name]
        expected_msg_count = message_total_count / participant_count
        user_ratio = user_count / expected_msg_count
        ratios.append(user_ratio)
        #print("%8d of which %s sent %5d" % (message_total_count, user_name, user_count))
    return ratios


def plot_ratios_for_user(ratios, user_name, fontsize=20, linewidth=2):
    counts, bins = np.histogram(ratios, bins=75, density=True)
    counts *= 100 / sum(counts)
    bwidth = bins[1] - bins[0]
    bins += bwidth / 2
    bins = bins[:-1]

    plt.bar(bins, counts, width=bwidth*0.8, label="%s's activity ratios"%user_name, color=(0, 0, 1, 0.5))
    plt.ylabel("%% of %s's conversations" % user_name, fontsize=fontsize)
    plt.xlabel("Activity ratio (1 means as active as 1 average member)", fontsize=fontsize)
    plt.grid(True)
    plt.gca().axvline(1, c="red", label="Average member activity", linewidth=linewidth)

    (mu, sigma) = norm.fit(ratios)
    y = norm.pdf(bins, mu, sigma)
    y *= 100 / sum(y)
    plt.plot(bins, y, label="Fit curve", c="green", linestyle="--", linewidth=linewidth)

    plt.legend(fontsize=fontsize)


def main(zip_file, user_name):
    counts = get_counts_from_zip(zip_file)
    ratios = get_ratios_from_counts(counts, user_name)
    plt.figure()
    plot_ratios_for_user(ratios, user_name)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("zip", help="path to a facebook-downloaded messages zip file")
    parser.add_argument("full_name", help="name of user for which activity will be plotted")
    args = parser.parse_args()

    main(args.zip, args.full_name)
