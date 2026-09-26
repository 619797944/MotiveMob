import pickle
import numpy as np
import scipy.stats
import math
import re
from math import sin, cos, asin, sqrt, radians
import pandas as pd

import argparse

def invert_dict(d):
    return {value: key for key, value in d.items()}


def load_pickle(file_path):
    """Utility function to load a pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


# Key: location name + latitude + longitude
# Value: ID in the city network
pos_map = load_pickle('./data/pos_map.pkl')

# Key: "location name + latitude + longitude" (to ensure uniqueness)
# Value: "location name + a unique ID for this location (same name locations get different IDs)"
loc_map = load_pickle('./data/global_loc_map.pkl')

# Key: location name, Value: category from foursquare
cat = load_pickle('./data/location_activity_map.pkl')
map_loc = invert_dict(loc_map)


TRAJECTORY_TIME_RE = re.compile(
    r"(?P<prefix>\bat\s+)"
    r"(?P<hour>\d{1,2}):(?P<minute>[0-5]\d)"
    r"(?::(?P<second>[0-5]\d))?"
)


def normalize_trajectory_times(traj):
    """Put trajectory timestamps on the 10-minute evaluation grid.

    Times in the last minute of an hour advance to the next hour, except for
    03:59, which remains inside the logical day by mapping to 03:50.
    All other timestamps are floored to their containing 10-minute interval.
    """

    def replace_time(match):
        hour = int(match.group("hour"))
        minute = int(match.group("minute"))

        if minute == 59:
            if hour == 3:
                normalized_hour = 3
                normalized_minute = 50
            else:
                normalized_hour = hour + 1
                normalized_minute = 0
        else:
            normalized_hour = hour
            normalized_minute = (minute // 10) * 10

        return (
            f'{match.group("prefix")}'
            f"{normalized_hour:02d}:{normalized_minute:02d}:00"
        )

    return TRAJECTORY_TIME_RE.sub(replace_time, str(traj))


def geodistance(lat1, lng1, lat2, lng2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000
    distance = round(distance / 1000, 3)
    return distance


def calculate_intervals_from_4am(times, interval=10, start_hour=4):
    start_sec = start_hour * 3600
    intervals = []

    for t in times:
        s = t.strip().strip(".")
        if s in ("24:00", "24:00:00"):
            sec = 24 * 3600
        else:
            parts = s.split(":")
            if len(parts) == 2:
                h, m = map(int, parts)
                sec = h * 3600 + m * 60
            else:
                h, m, ss = map(int, parts)
                sec = h * 3600 + m * 60 + ss

        # day boundary is 04:00 -> next day 04:00: a time in 00:00-03:59 belongs to the previous day (+24h)
        if sec < start_sec:
            sec += 24 * 3600

        diff_min = (sec - start_sec) // 60
        intervals.append(diff_min // interval)

    return intervals

def clean_traj(traj):
    acts = traj.split(": ")[-1]
    return acts


def duration(p):
    d = [[i[0] - u[index][0] for index, i in enumerate(u[1:])] for u in p]
    d = [round(i * 10) for u in d for i in u]
    return d

def obtain_analysis_traj(data, pos_map=None, map_loc=None, cat=None):
    # None defaults (not the module-level dicts directly) so help()/tracebacks/docs
    # don't try to repr a ~300k-entry dict as part of the function signature.
    if pos_map is None:
        pos_map = globals()["pos_map"]
    if map_loc is None:
        map_loc = globals()["map_loc"]
    if cat is None:
        cat = globals()["cat"]

    traj_ids = []
    traj_lat_lngs = []
    traj_act_ts = []
    wrong = 0

    UNKNOWN_ACT_ID = p2id["Unknown"]

    for d, traj in data.items():
        traj = data[d]
        traj = normalize_trajectory_times(traj)
        if ": : " in traj:
            traj = traj.replace(": : ", ": ")
        if " :" in traj:
            traj = traj.replace(", ", "")

        traj_acts = clean_traj(traj)

        buffer = ""
        steps = []
        for p in traj_acts.split(","):
            if "00" in p.strip():
                p = buffer + p.strip()
                steps.append(p.strip())
                buffer = ""
            else:
                buffer = buffer + p.strip() + ", "

        loc_times = []
        for st in steps:
            loc_times.extend(st.split(" at "))

        locs = []
        times = []
        acts = []

        k = 0
        while k < len(loc_times):
            loc_times[k] = loc_times[k].replace(".", "")

            if k % 2 == 0:
                clean_loc = loc_times[k]

                if "Home" in clean_loc or "home" in clean_loc:
                    k += 2
                    continue

                loc_key = clean_loc.split("#")[0].strip()
                act_id = cat.get(loc_key, UNKNOWN_ACT_ID)  # unknown categories fall into "Unknown"
                acts.append(act_id)
                locs.append(clean_loc)

            else:
                if ':' in loc_times[k].split(" ")[0]:
                    times.append(loc_times[k].split(" ")[0])
                else:
                    acts.pop()
                    locs.pop()

            k += 1

        times_interval = calculate_intervals_from_4am(times)

        traj_id, traj_lat_lng, traj_act_t = [], [], []

        for i in range(len(locs)):
            if "Home" in locs[i] or "home" in locs[i]:
                continue

            try:
                loc_with_lat_lng = map_loc[locs[i].strip()]
            except:
                print(locs[i])
                wrong += 1
                continue

            loc_with_lat_lng_ = loc_with_lat_lng.replace(" (", ", ").replace(")", "")
            lat_lng = [float(loc_with_lat_lng_.split(", ")[-2]), float(loc_with_lat_lng_.split(", ")[-1])]
            loc_id = pos_map[loc_with_lat_lng]
            t = times_interval[i]

            traj_id.append([loc_id, t])
            traj_act_t.append([acts[i], t])
            traj_lat_lng.append([lat_lng[0], lat_lng[1], t])

        traj_ids.append(traj_id)
        traj_act_ts.append(traj_act_t)
        traj_lat_lngs.append(traj_lat_lng)

    return traj_ids, traj_lat_lngs, traj_act_ts, wrong


p2id = {'Travel and Transportation': 0, 'Dining and Drinking': 1, 'Retail': 2,
        'Community and Government': 3, 'Arts and Entertainment': 4, 'Business and Professional Services': 5,
        'Landmarks and Outdoors': 6,
        'Health and Medicine': 7, 'Sports and Recreation': 8, 'Event': 9}
p2id["Unknown"] = len(p2id)


def activity_to_id(activity):
    if isinstance(activity, (int, np.integer)):
        activity = int(activity)
        return activity if activity in set(p2id.values()) else p2id["Unknown"]
    return p2id.get(activity, p2id["Unknown"])


def transfer(data):
    transfer_data = []
    locs_id = data[0]
    lat_lngs = data[1]
    acts = data[2]
    for i in range(len(locs_id)):
        this_day = []
        for j in range(len(lat_lngs[i])):
            this_day.append([locs_id[i][j][1], activity_to_id(acts[i][j][0]), [lat_lngs[i][j][0], lat_lngs[i][j][1]]])
        sorted_this_day = sorted(this_day, key=lambda x: x[0])
        transfer_data.append(sorted_this_day)
    return transfer_data


class Evaluation(object):
    def __init__(self, args):
        self.args = args

    def arr_to_distribution(self, arr, Min, Max, bins):
        distribution, base = np.histogram(arr[arr <= Max], bins=bins, range=(Min, Max))
        m = np.array([len(arr[arr > Max])], dtype='int64')
        distribution = np.hstack((distribution, m))
        return distribution, base[:-1]

    def get_js_divergence(self, p1, p2):
        p1 = p1 / (p1.sum() + 1e-9)
        p2 = p2 / (p2.sum() + 1e-9)
        m = (p1 + p2) / 2
        js = 0.5 * scipy.stats.entropy(p1, m) + 0.5 * scipy.stats.entropy(p2, m)
        return js

    def distance_one_step(self, p1, p2):
        f = [geodistance(i[2][0], i[2][1], u[index][2][0], u[index][2][1]) for u in p1 for index, i in enumerate(u[1:])]
        r = [geodistance(i[2][0], i[2][1], u[index][2][0], u[index][2][1]) for u in p2 for index, i in enumerate(u[1:])]
        MIN = 0
        MAX = 50
        bins = int(math.ceil(MAX - MIN)/1)
        r_list, _ = self.arr_to_distribution(np.array(r), MIN, MAX, bins)
        f_list, _ = self.arr_to_distribution(np.array(f), MIN, MAX, bins)
        r_list = r_list / (r_list.sum() + 1e-9)
        f_list = f_list / (f_list.sum() + 1e-9)
        JSD = self.get_js_divergence(r_list, f_list)
        return JSD

    def st_act_jsd(self, p1, p2):
        st_act_dict = {}
        for u in p1:
            for i in u:
                if str(i[0]) + '_' + str(i[1]) not in st_act_dict:
                    st_act_dict[str(i[0]) + '_' + str(i[1])] = len(st_act_dict)
        for u in p2:
            for i in u:
                if str(i[0]) + '_' + str(i[1]) not in st_act_dict:
                    st_act_dict[str(i[0]) + '_' + str(i[1])] = len(st_act_dict)
        f, r = [], []
        for u in p1:
            for i in u:
                f.append(st_act_dict[str(i[0]) + '_' + str(i[1])])
        for u in p2:
            for i in u:
                r.append(st_act_dict[str(i[0]) + '_' + str(i[1])])
        MIN = np.min(r + f)
        MAX = np.max(r + f)
        bins = 1000
        r = (np.array(r) - MIN) / (MAX - MIN)
        f = (np.array(f) - MIN) / (MAX - MIN)
        r_list, _ = self.arr_to_distribution(r, 0, 1, bins)
        f_list, _ = self.arr_to_distribution(f, 0, 1, bins)
        JSD = self.get_js_divergence(r_list, f_list)
        return JSD

    def st_loc_jsd(self, p1, p2):
        st_act_dict = {}
        for u in p1:
            for i in u:
                if str(i[0]) + '_' + str(i[2][0]) + '_' + str(i[2][1]) not in st_act_dict:
                    st_act_dict[str(i[0]) + '_' + str(i[2][0]) + '_' + str(i[2][1])] = len(st_act_dict)
        for u in p2:
            for i in u:
                if str(i[0]) + '_' + str(i[2][0]) + '_' + str(i[2][1]) not in st_act_dict:
                    st_act_dict[str(i[0]) + '_' + str(i[2][0]) + '_' + str(i[2][1])] = len(st_act_dict)
        f, r = [], []
        for u in p1:
            for i in u:
                f.append(st_act_dict[str(i[0]) + '_' + str(i[2][0]) + '_' + str(i[2][1])])
        for u in p2:
            for i in u:
                r.append(st_act_dict[str(i[0]) + '_' + str(i[2][0]) + '_' + str(i[2][1])])
        MIN = np.min(r + f)
        MAX = np.max(r + f)
        bins = 400
        r = (np.array(r) - MIN) / (MAX - MIN)
        f = (np.array(f) - MIN) / (MAX - MIN)
        r_list, _ = self.arr_to_distribution(np.array(r), 0, 1, bins)
        f_list, _ = self.arr_to_distribution(np.array(f), 0, 1, bins)
        JSD = self.get_js_divergence(r_list, f_list)
        return JSD

    def duration_jsd(self, p1, p2):
        f = duration(p1)
        r = duration(p2)
        MIN = 0
        MAX = 2400
        bins = math.ceil(MAX - MIN)//10
        r_list, _ = self.arr_to_distribution(np.array(r), MIN, MAX, bins)
        f_list, _ = self.arr_to_distribution(np.array(f), MIN, MAX, bins)
        JSD = self.get_js_divergence(r_list, f_list)
        return JSD

    # return order maps to the printed metric names: duration_jsd->SI, distance_step->SD, st_act_jsd->DARD, st_loc_jsd->STVD
    def get_JSD(self, real, fake):
        duration_jsd = self.duration_jsd(real, fake)
        distance_step = self.distance_one_step(real, fake)
        st_act_jsd = self.st_act_jsd(real, fake)
        st_loc_jsd = self.st_loc_jsd(real, fake)
        return duration_jsd, distance_step, st_act_jsd, st_loc_jsd


def eval_llm(csv_path, pos_map=None, map_loc=None, cat=None):
    df = pd.read_csv(csv_path)

    required_cols = {"person", "real_traj", "gen_traj"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"inference_results.csv 缺少列: {missing}. 需要 {required_cols}")

    df = df.dropna(subset=["person", "real_traj", "gen_traj"]).copy()
    df["person"] = df["person"].astype(str)

    evaluation = Evaluation(None)

    all_real_data = []
    all_gen_data = []
    errors = 0
    all_bad = []
    total_days = 0
    valid_persons = 0

    for person, g in df.groupby("person"):
        # row index as key: only needs to be unique, doesn't need to mean anything
        real_dict = {f"{person}_{idx}": traj for idx, traj in zip(g.index, g["real_traj"])}
        gen_dict  = {f"{person}_{idx}": traj for idx, traj in zip(g.index, g["gen_traj"])}

        real_traj_ids, real_latlngs, real_acts, _ = obtain_analysis_traj(real_dict, pos_map=pos_map, map_loc=map_loc, cat=cat)
        gen_traj_ids,  gen_latlngs,  gen_acts, wrong  = obtain_analysis_traj(gen_dict, pos_map=pos_map, map_loc=map_loc, cat=cat)
        errors+=wrong
        real_data = transfer([real_traj_ids, real_latlngs, real_acts])
        gen_data  = transfer([gen_traj_ids,  gen_latlngs,  gen_acts])
        bad = 0
        for gen in gen_data:
            if len(gen) <= 0:
                bad += 1
        all_bad.append(bad)
        total_days += len(gen_data)

        # some persons may end up empty if trajectory parsing failed or everything got filtered out
        if len(real_data) == 0 or len(gen_data) == 0:
            print(f"[WARN] person={person} real_data/gen_data 为空，跳过（可能轨迹解析失败或都被过滤掉了）")
            continue

        valid_persons += 1
        all_real_data.extend(real_data)
        all_gen_data.extend(gen_data)

    if valid_persons == 0:
        print("没有任何 person 产生可评估的数据（检查 real_traj/gen_traj 格式、cat/loc_map/pos_map 是否覆盖）。")
        return

    # errors: total locations obtain_analysis_traj() couldn't map to lat/lng, summed across all persons
    print(errors)
    # aggregate every person's days into one pool instead of averaging per-person JSDs
    duration_jsd, distance_step, st_act_jsd, st_loc_jsd = evaluation.get_JSD(all_real_data, all_gen_data)
    print("\nOverall:")
    print(
        f"SD: {np.mean(distance_step):.4f}, "
        f"SI: {np.mean(duration_jsd):.4f}, "
        f"DARD: {np.mean(st_act_jsd):.4f}, "
        f"STVD: {np.mean(st_loc_jsd):.4f}"
    )
    print(f"{sum(all_bad)} days out of {total_days} days length less than 2, the rate is{sum(all_bad)/total_days}")
    print()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="merged.csv")
    args = ap.parse_args()
    eval_llm(args.input)
