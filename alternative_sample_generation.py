from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

NUM_TIME_BINS = 144
DAY_START_MIN = 4 * 60  # 04:00 -> 240 minutes

MOTIVATION_LABELS = [
    "Return", "Personal Service", "Entertainment", "Dining", "Transit",
    "Religious", "Education", "Work", "Social", "Shopping", "Healthcare",
]


def invert_dict(d):
    return {value: key for key, value in d.items()}


def is_weekend(date):
    return int(date.weekday() >= 5)


mot2id = {name: idx for idx, name in enumerate(MOTIVATION_LABELS)}

# Populated by _main_gen_data() before any of the construct_df_* functions run.
loc_map = {}
map_loc = {}
subcat2id = {}
id2subcat = {}
activity_map = {}


DG_PKL_DIR = Path("./seen_users")

def get_lat_lng_llm(loc_name):
    pos = loc_name.split('(')[-1].split(')')[0]
    
    lat_str, lng_str = [x.strip() for x in pos.split(',')]
    lat = float(lat_str)
    lng = float(lng_str)
    return lat, lng

def parse_user_pkl_llm(obj: Any) -> Tuple[List[str], Any, str]:
    # returns (trajectories, motivations, behavior_report)
    # See parse_user_pkl(): current on-disk pkls are 5 items (a 5th, COVID-context
    # report, follows the plain one); older 4-item pkls still fall back correctly.
    if isinstance(obj, list) and len(obj) == 5:
        trajs, test, motos, report, _report_covid = obj
        return trajs, motos, report
    if isinstance(obj, list) and len(obj) == 4:
        trajs, test, motos, report = obj
        return trajs, motos, report
    raise ValueError("Unsupported pkl structure; expected dict or (trajs, motos, report).")

def extract_date_and_steps_llm(traj: str) -> Tuple[Optional[str], List[str]]:
    # "Activities at 2019-01-05: A at 22:40:00, B at 01:30:00"
    # returns ("2019-01-05", ["A at 22:40:00", "B at 01:30:00"])
    s = " ".join(traj.split())
    date = None
    try:
        # grab date token after "at"
        prefix = s.split(":", 1)[0]  # "Activities at 2019-01-05"
        date = prefix.split()[-1]
    except Exception:
        date = None
    after = s.split(":", 1)[1].strip() if ":" in s else s.strip()
    after = after.rstrip(".")
    buffer = ""
    steps = []
    for p in after.split(","):
        if "00" in p.strip():
            p = buffer + p.strip()
            steps.append(p.strip())
            buffer = ""
        else:
            buffer = buffer + p.strip() + ", "
    return date, steps

def time_str_to_bin_llm(t_str: str) -> int:
    """
    t_str: 'HH:MM:SS'
    Returns a time-bin index in 0~143.
    Convention: bin 0 = 04:00:00, bin 143 = 03:50:00 the next day.
    """
    h, m, s = map(int, t_str.split(':'))
    total_min = h * 60 + m  # minutes since 00:00 (0 ~ 1439)

    # Treat 4:00 as the zero point: shift by -240 minutes, then mod 1440 to avoid negatives.
    shifted = (total_min - DAY_START_MIN) % (24 * 60)

    bin_idx = shifted // 10  # one bin per 10 minutes
    return int(bin_idx)

def bin_to_time_str_llm(bin_idx: int) -> str:
    """
    0 -> '04:00:00'
    1 -> '04:10:00'
    ...
    143 -> '03:50:00' (early morning of the next day)
    """
    if not (0 <= bin_idx < NUM_TIME_BINS):
        raise ValueError(f"bin_idx should be in [0, {NUM_TIME_BINS-1}], got {bin_idx}")

    # Compute minutes since 4:00 first, then shift back onto the normal time axis.
    shifted_min = bin_idx * 10                       # relative to 4:00
    total_min = (shifted_min + DAY_START_MIN) % (24 * 60)

    h = total_min // 60
    m = total_min % 60
    return f"{h:02d}:{m:02d}:00"

def classify_motivation(to_poi):
    return_list = {
        "Hotel", "Bed and Breakfast", "Hostel", "Resort", "Inn",
        "Boarding House", "Lodging", "Motel", "Vacation Rental", "RV Park"
    }
    personal_service_list = {
        "Fuel Station", "Electric Vehicle Charging Station", 
        "Rental Car Location", "Bike Rental", "Boat Rental", "Baggage Locker",
        "Tourist Information and Service", "Travel Agency", "Tour Provider"
    }
    religious = {
        'Shrine', 'Buddhist Temple', 'Spiritual Center', 'Church', 'Mosque',
        'Temple', 'Cemevi', 'Confucian Temple', 'Sikh Temple', 'Prayer Room',
        'Hindu Temple', 'Monastery', 'Kingdom Hall', 'Synagogue'
    }
    education = {
        'Music School', 'University', 'Community College', 'Middle School',
        'Elementary School', 'High School', 'College and University',
        'College Academic Building', 'Trade School',
        'Driving School', 'Library', 'College Engineering Building',
        'Preschool', 'Adult Education', 'Private School', 'Religious School',
        'Medical School', 'College Library',
        'College Classroom', 'College Gym', 'College Arts Building',
        'Primary and Secondary School', 'College Quad', 'College Bookstore',
        'College Lab', 'Language School', 'College Science Building',
        'College Auditorium', 'Nursery School', 'Culinary School',
        'College Theater', 'College Stadium', 'College Technology Building',
        'College Communications Building', 'College Rec Center',
        'Summer Camp', 'College Tennis Court', 'Law School', 'College Soccer Field',
        'College Baseball Diamond', 'College Football Field', 'College Track',
        'Circus School', 'College Math Building', 'Observatory', 'Education', 'Flight School',
        'Computer Training School'
    }
    return_home = {
        'Apartment or Condo', 'College Residence Hall', 'Home (private)',
        'Housing Development', 'Residential Building', 'Trailer Park', "Assisted Living"
    }
    work = {
        'Police Station', 'Government Building', 'City Hall', 'Fire Station',
        'Courthouse', 'Town Hall', 'Military Base', 'Capitol Building',
        'Embassy or Consulate', 'Non-Profit Organization',
        'Social Services Organization', 'Public and Social Service',
        'Labor Union', 'Probation Office', 'Polling Place', 'Rescue Service',
        'Organization', 'Prison', 'College Administrative Building'
    }
    personal_service = {
        "Spa", "Massage Clinic", "Health and Beauty Service", "Nail Salon",
        "Hair Salon", "Bath House", "Repair Service", "Automotive Repair Shop",
        "Laundromat", "ATM", "Tire Repair Shop", "Barbershop", "Car Wash and Detail",
        "Photography Service", "Rental Service", "Pet Service", "Design Studio",
        "Dry Cleaner", "Insurance Agency", "Mover", "Laundry Service",
        "Credit Union", "Shoe Repair Service", "Machine Shop", "Tailor",
        "Child Care Service", "Tattoo Parlor", "Skin Care Clinic",
        "Locksmith", "Tanning Salon", "Body Piercing Shop", "Notary",
        "Check Cashing Service", "Computer Repair Service",
        "Automotive Service", "Motorcycle Repair Shop", "Skin Care Clinic", "Daycare",
        "Financial Service", "Bank", "Lottery Retailer", "Photography Lab", "Waste Management Service",
        "Home Service", "Event Service", "Storage Facility", "Shipping, Freight, and Material Transportation Service",
        "Employment Agency", "Vehicle Inspection Station", "IT Service", "Currency Exchange",
        "Water Treatment Service", "Agriculture and Forestry Service", "Telecommunication Service",
        "Real Estate Service", "Import and Export Service", "Professional Cleaning Service", "Equipment Rental Service",
        "Painter", "Legal Service", "Tutoring Service"
    }
    if to_poi[5] == "Travel and Transportation":
        if to_poi[6] in return_list:
            return "Return"
        elif to_poi[6] in personal_service_list:
            return "Personal Service"
        elif to_poi[6] in ["Cruise", "Hotel Pool"]:
            return "Entertainment"
        elif to_poi[6] == "Airport Food Court":
            return "Dining"
        else:
            return "Transit"
    elif to_poi[5] == "Community and Government":
        if to_poi[6] in religious:
            return "Religious"
        elif to_poi[6] in education:
            return "Education"
        elif to_poi[6] in return_home:
            return "Return"
        elif to_poi[6] in work:
            return "Work"
        elif to_poi[6] in ['Post Office', 'Public Bathroom', 'Animal Shelter',
        'Rehabilitation Center']:
            return "Personal Service"
        elif to_poi[6] == "College Cafeteria":
            return "Dining"
        else:
            return "Social"
    elif to_poi[5] == "Business and Professional Services":
        if to_poi[6] in personal_service:
            return "Personal Service"
        elif to_poi[6] in ["Wedding Hall", "Funeral Home"]:
            return "Social"
        elif to_poi[6] in ["Auditorium","Ballroom","Print, TV, Radio and Outdoor Advertising Service", "Entertainment Service"]:
            return "Entertainment"
        elif to_poi[6] == "Food and Beverage Service":
            return "Shopping"
        else:
            return "Work"
    elif to_poi[5] == "Dining and Drinking":
        return "Dining"
    elif to_poi[5] == "Arts and Entertainment":
        return "Entertainment"
    elif to_poi[5] == "Retail":
        return "Shopping"
    elif to_poi[5] == "Health and Medicine":
        return "Healthcare"
    elif to_poi[5] == "Landmarks and Outdoors":
        return "Entertainment"
    elif to_poi[5] == "Sports and Recreation":
        return "Entertainment"
    elif to_poi[5] == "Event":
        return "Social"

def extract_prediction(val):
    if pd.isna(val):
        return None
    m = re.search(r'"prediction"\s*:\s*"([^"]+)"', str(val))
    return m.group(1) if m else val

HISTORY_DAYS = 3

def build_traj_by_date(trajectories) -> Dict[pd.Timestamp, List[str]]:
    """One user's trajectories -> {date: steps}, for looking back at earlier days."""
    traj_by_date = {}
    for traj in trajectories:
        date_str, steps = extract_date_and_steps_llm(traj)
        if not date_str or not steps:
            continue
        traj_by_date[pd.to_datetime(date_str)] = steps
    return traj_by_date

def build_recent_history(traj_by_date: Dict[pd.Timestamp, List[str]], cur_date, days: int = HISTORY_DAYS) -> str:
    """Join the most recent `days` days strictly before cur_date (prompt_builder.py's
    expected "Activities at <date>: <step>, <step>, ..." per line, newline-joined).
    Empty string if there are no earlier days."""
    cur_date = pd.to_datetime(cur_date)
    prev_dates = sorted(d for d in traj_by_date if d < cur_date)[-days:]
    return "\n".join(
        f"Activities at {d.strftime('%Y-%m-%d')}: {', '.join(traj_by_date[d])}"
        for d in prev_dates
    )

def _resolve_person_path(person, pkl_dir: Path) -> Path:
    p = Path(person)
    if p.is_file():
        return p
    candidate = pkl_dir / p.name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Cannot find pkl for person={person!r} (tried {p} and {candidate})")

def history_lookup_for_persons(persons: List[str], dates: List, pkl_dir: Path, days: int = HISTORY_DAYS) -> List[str]:
    """Row-wise history for rows that only carry a person path + date (e.g. rows
    read back from a judge/llm_judge CSV) - opens each unique person's pkl once."""
    cache: Dict[str, Dict[pd.Timestamp, List[str]]] = {}
    histories = []
    for person, date in zip(persons, dates):
        if person not in cache:
            try:
                pkl_path = _resolve_person_path(person, pkl_dir)
                with open(pkl_path, "rb") as f:
                    obj = pickle.load(f)
                trajectories, _motivations, _report = parse_user_pkl_llm(obj)
                cache[person] = build_traj_by_date(trajectories)
            except Exception:
                cache[person] = {}
        histories.append(build_recent_history(cache[person], date, days=days))
    return histories

def construct_llm_df(
    input_csv="prompt_trained_alt_predictions_4000.csv",  # change to the llama8 output filename
    text_col="llama_prompt_predictions",
    pkl_dir: Path = DG_PKL_DIR,
):
    output_columns = [
        "person",
        "original",
        "date",
        "year",
        "month",
        "day",
        "is_weekend",
        "motivation_id",
        "llm_text",
        "history",
    ]

    # ============================================================
    # 1. Read the llama8_prompt_judge output and map its columns
    # ============================================================
    old_df = pd.read_csv(input_csv)

    # Column mapping: llama8/transformer judge output -> output_columns.
    # Some newer judge outputs already contain both "current_checkin" and
    # "original". Renaming current_checkin blindly would create duplicate
    # "original" columns and make pd.concat fail with InvalidIndexError.
    if "current_checkin" in old_df.columns:
        old_df["original"] = old_df["current_checkin"]
    if text_col in old_df.columns:
        old_df["llm_text"] = old_df[text_col]

    # Remove duplicate columns defensively in case an older input was already
    # saved with non-unique headers.
    old_df = old_df.loc[:, ~old_df.columns.duplicated(keep="first")]

    old_df["llm_text"] = old_df["llm_text"].apply(extract_prediction)

    # date is a string in the CSV; re-parse it and derive year/month/day
    old_df["date"]  = pd.to_datetime(old_df["date"])
    old_df["year"]  = old_df["date"].dt.year
    old_df["month"] = old_df["date"].dt.month
    old_df["day"]   = old_df["date"].dt.day

    # Drop duplicate-named columns (the original column name may coexist with
    # the renamed target column)
    # old_df = old_df.loc[:, ~old_df.columns.duplicated(keep="first")]

    # Fill in any missing columns, then keep only the ones we need
    for c in output_columns:
        if c not in old_df.columns:
            old_df[c] = None

    old_df["person"] = old_df["person"].astype(str)
    old_df["history"] = history_lookup_for_persons(old_df["person"].tolist(), old_df["date"].tolist(), pkl_dir)

    old_df = old_df[output_columns]
    old_df = old_df.dropna(subset=["llm_text"])  # drop rows with no correct answer

    print("Loaded prediction rows:", len(old_df))

    # ============================================================
    # 2. Generate new START rows - use every PKL directly under pkl_dir
    #    (e.g. seen_users), no sample-file whitelist.
    # ============================================================
    sample_set = sorted(pkl_dir.rglob("*.pkl"))
    if not sample_set:
        raise FileNotFoundError(f"No PKLs found in {pkl_dir}")

    print(sample_set)
    print("Found pkls:", len(sample_set))

    rows = []

    for p in sample_set:
        with open(p, "rb") as f:
            obj = pickle.load(f)

        trajectories, motivations, report = parse_user_pkl_llm(obj)
        traj_by_date = build_traj_by_date(trajectories)

        for ti, traj in enumerate(trajectories):
            date, steps = extract_date_and_steps_llm(traj)

            if len(steps) < 1:
                continue

            date = pd.to_datetime(date)
            year = date.year
            month = date.month
            day = date.day
            weekend_flag = is_weekend(date)

            cur_loc_id, cur_time = steps[0].split(" at ")
            cur_loc, cur_id = (
                cur_loc_id.strip().split("#")[0],
                cur_loc_id.strip().split("#")[1],
            )

            raw_mot = [0, 0, 0, 0, 0, activity_map[cur_loc], cur_loc]
            motivation = classify_motivation(raw_mot)

            row = {
                "person": p,
                "original": "START",
                "date": date,
                "year": year,
                "month": month,
                "day": day,
                "is_weekend": weekend_flag,
                "motivation_id": mot2id[motivation],
                "llm_text": steps[0],
                "history": build_recent_history(traj_by_date, date),
            }

            rows.append(row)

    new_df = pd.DataFrame(rows)

    print("New START rows:", len(new_df))

    if len(new_df) > 0:
        new_df = new_df.dropna(how="any")
        new_df["date"] = pd.to_datetime(new_df["date"])

    print("New START rows after drop:", len(new_df))

    # ============================================================
    # 3. Align the newly generated data with the old CSV's columns
    # ============================================================

    # Cast person to str to avoid mixed types between parquet/csv
    if "person" in new_df.columns:
        new_df["person"] = new_df["person"].astype(str)

    if "person" in old_df.columns:
        old_df["person"] = old_df["person"].astype(str)

    # Fill in every output_columns column on new_df
    for c in output_columns:
        if c not in new_df.columns:
            new_df[c] = None

    # Keep only the old CSV's schema columns
    new_df = new_df[output_columns]

    # Keep old_df in the same column order too
    old_df = old_df[output_columns]



    # ============================================================
    # 4. Concatenate the old prediction rows with the new START rows
    # ============================================================
    df_out = pd.concat([old_df, new_df], ignore_index=True)

    print("Old rows:", len(old_df))
    print("New START rows:", len(new_df))
    print("Total rows after concat:", len(df_out))

    # ============================================================
    # 5. Save
    # ============================================================
    # df_out.to_csv(output_csv, index=False)

    # print(f"Saved merged data to {output_csv}")

    return df_out

def construct_df_pos():
    # Use every PKL directly under DG_PKL_DIR (e.g. seen_users), no sample-file whitelist.
    sample_set = sorted(DG_PKL_DIR.rglob("*.pkl"))
    print(sample_set)
    print("Found pkls:", len(sample_set))

    rows = []

    for p in sample_set:
        with open(p, "rb") as f:
            obj = pickle.load(f)

        trajectories, motivations, report = parse_user_pkl_llm(obj)
        traj_by_date = build_traj_by_date(trajectories)

        for ti, traj in enumerate(trajectories):
            date, steps = extract_date_and_steps_llm(traj)
            if len(steps) < 2:
                continue

            date = pd.to_datetime(date)
            year = date.year
            month = date.month
            day = date.day
            weekend_flag = is_weekend(date)
            history = build_recent_history(traj_by_date, date)
            cur_loc_id, cur_time = steps[0].split(' at ')
            cur_loc, cur_id = cur_loc_id.strip().split("#")[0], cur_loc_id.strip().split("#")[1]
            raw_mot = [0, 0, 0, 0, 0, activity_map[cur_loc], cur_loc]
            motivation = classify_motivation(raw_mot)

            row = {
                "person": p,
                "original": "START",
                "date": date,
                "year": year,
                "month": month,
                "day": day,
                "is_weekend": weekend_flag,

                "motivation_id": mot2id[motivation],
                "llm_text": steps[0],
                "history": history,
            }

            rows.append(row)
            for i in range(len(steps) - 1):
                motivation = mot2id[motivations[ti][i]]

                row = {
                    "person": p,
                    "original": steps[i],
                    "date": date,
                    "year": year,
                    "month": month,
                    "day": day,
                    "is_weekend": weekend_flag,

                    "motivation_id": motivation,
                    "llm_text": steps[i+1],
                    "history": history,
                }

                rows.append(row)

    df = pd.DataFrame(rows)
    print("Total transitions:", len(df))
    df = df.dropna(how="any")
    print("Total transitions after drop:", len(df))
    # Ensure date is a datetime type
    df["date"] = pd.to_datetime(df["date"])

    return df

def construct_df_phase2():
    # Use every PKL directly under DG_PKL_DIR (e.g. seen_users), no sample-file whitelist.
    sample_set = sorted(DG_PKL_DIR.rglob("*.pkl"))
    print(sample_set)
    print("Found pkls:", len(sample_set))

    rows = []

    for p in sample_set:
        with open(p, "rb") as f:
            obj = pickle.load(f)

        trajectories, motivations, report = parse_user_pkl_llm(obj)
        traj_by_date = build_traj_by_date(trajectories)

        for ti, traj in enumerate(trajectories):
            date, steps = extract_date_and_steps_llm(traj)
            if len(steps) < 2:
                continue

            date = pd.to_datetime(date)
            year = date.year
            month = date.month
            day = date.day
            weekend_flag = is_weekend(date)
            history = build_recent_history(traj_by_date, date)
            cur_loc_id, cur_time = steps[0].split(' at ')
            cur_loc, cur_id = cur_loc_id.strip().split("#")[0], cur_loc_id.strip().split("#")[1]
            raw_mot = [0, 0, 0, 0, 0, activity_map[cur_loc], cur_loc]
            motivation = classify_motivation(raw_mot)
            row = {
                "person": p,
                "original": "START",
                "date": date,
                "year": year,
                "month": month,
                "day": day,
                "is_weekend": weekend_flag,

                "motivation": motivation,
                "history": history,
            }
            rows.append(row)
            for i in range(len(steps) - 1):
                cur_loc_id, cur_time = steps[i].split(' at ')
                next_loc_id, next_time = steps[i + 1].split(' at ')
                cur_time_bin = time_str_to_bin_llm(cur_time)
                next_time_bin = time_str_to_bin_llm(next_time)
                cur_loc, cur_id = cur_loc_id.strip().split("#")[0], cur_loc_id.strip().split("#")[1]
                cur_loc = subcat2id[cur_loc.strip()]
                next_loc, next_id = next_loc_id.strip().split("#")[0], next_loc_id.strip().split("#")[1]
                next_loc = subcat2id[next_loc.strip()]
                motivation = motivations[ti][i]


                row = {
                    "person": p,
                    "original": steps[i],
                    "date": date,
                    "year": year,
                    "month": month,
                    "day": day,
                    "is_weekend": weekend_flag,

                    "cur_loc_name": cur_loc,
                    "cur_time": cur_time,
                    "cur_subcat_id": cur_loc,
                    "cur_time_bin": cur_time_bin,
                    "cur_id": cur_id,

                    "next_subcat_id": next_loc,
                    "next_time_bin": next_time_bin,
                    "next_id": next_id,

                    "motivation": motivation,
                    "history": history,
                }
                rows.append(row)
            # Add END row
            row = {
                    "person": p,
                    "original": steps[-1],
                    "date": date,
                    "year": year,
                    "month": month,
                    "day": day,
                    "is_weekend": weekend_flag,

                    "motivation": "END",
                    "history": history,
                }
            rows.append(row)

    df = pd.DataFrame(rows)
    print("Total transitions:", len(df))
    required_cols = ["person", "original", "date", "is_weekend", "motivation"]
    df = df.dropna(subset=required_cols)
    print("Total transitions after drop:", len(df))
    print("START rows:", int(df["original"].astype(str).str.strip().eq("START").sum()))
    print("END rows:", int(df["motivation"].astype(str).str.strip().eq("END").sum()))
    # Ensure date is a datetime type
    df["date"] = pd.to_datetime(df["date"])

    return df

def _run_gen_data(args):
    global DG_PKL_DIR, loc_map, map_loc, activity_map, subcat2id, id2subcat

    DG_PKL_DIR = args.pkl_dir
    with args.location_map.open("rb") as handle:
        loc_map = pickle.load(handle)
    with args.location_activity_map.open("rb") as handle:
        activity_map = pickle.load(handle)
    map_loc = invert_dict(loc_map)
    subcategory_names = sorted(
        {
            str(location_id).split("#", 1)[0].strip()
            for location_id in map_loc
        }
    )
    subcat2id = {
        name: index
        for index, name in enumerate(subcategory_names)
    }
    id2subcat = invert_dict(subcat2id)
    if args.phase == "mot":
        df = construct_df_phase2()
        out_path_csv = args.output_csv or "motivation_training_data.csv"

        df.to_csv(out_path_csv, index=False)
        print(f"Saved motivation predictor training data to {out_path_csv}")
    else:
        # state: train the state predictor (implicit_world_modeling.py) - builds
        # both halves it needs in one call. Mix them yourself afterward:
        #   alternative_sample_generation.py mix --pos state_training_observed_data.csv \
        #       --neg state_training_alternative_data.csv --out state_training_data.csv
        df_observed = construct_df_pos()
        out_observed_csv = args.output_observed_csv or "state_training_observed_data.csv"
        df_observed.to_csv(out_observed_csv, index=False)
        print(f"Saved observed (real) state training data to {out_observed_csv}")

        df_alternative = construct_llm_df(
            input_csv=args.input_csv,
            text_col="llm_judge_text",
            pkl_dir=args.pkl_dir,
        )
        out_alternative_csv = args.output_csv or "state_training_alternative_data.csv"
        df_alternative.to_csv(out_alternative_csv, index=False)
        print(f"Saved alternative (counterfactual) state training data to {out_alternative_csv}")

# ==========================================================================
# 3. Positive/negative mixing (from mix_pos_neg.py)
# ==========================================================================

MIX_REQUIRED = ["person", "original", "motivation_id", "llm_text", "date", "is_weekend", "history"]

def mix_ensure_cols(df: pd.DataFrame, name: str):
    miss = [c for c in MIX_REQUIRED if c not in df.columns]
    if miss:
        raise ValueError(f"[{name}] missing columns: {miss}")

def _run_mix(args):
    pos = pd.read_csv(args.pos)
    neg = pd.read_csv(args.neg)

    mix_ensure_cols(pos, "pos")
    mix_ensure_cols(neg, "neg")

    # Keep only the columns training actually uses, so stray fields don't cause problems
    pos = pos[MIX_REQUIRED].copy()
    neg = neg[MIX_REQUIRED].copy()

    # An optional marker column to help debugging later (training code ignores it)
    pos["is_pos"] = 1
    neg["is_pos"] = 0

    # Decide how many rows to sample
    r = args.pos_ratio
    if r < 0 or r > 1:
        raise ValueError("pos_ratio must be in [0,1]")

    if args.total is None or args.total < 0:
        # Use "as much data as possible": take the largest value achievable at this pos_ratio (no resampling)
        max_pos = len(pos)
        max_neg = len(neg)
        # Keep pos / (pos+neg) ~= pos_ratio
        # pos = rT, neg=(1-r)T => T <= min(max_pos/r, max_neg/(1-r))
        if r == 0:
            T = max_neg
        elif r == 1:
            T = max_pos
        else:
            T = int(min(max_pos / r, max_neg / (1 - r)))
    else:
        T = int(args.total)

    n_pos = int(round(T * r))
    n_neg = T - n_pos

    pos_s = pos.sample(n=min(n_pos, len(pos)), replace=False, random_state=args.seed)
    neg_s = neg.sample(n=min(n_neg, len(neg)), replace=False, random_state=args.seed)

    combined = pd.concat([pos_s, neg_s], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    combined.to_csv(args.out, index=False)
    print(f"[OK] wrote {len(combined)} rows to {args.out}")
    print(f"      pos={len(pos_s)} neg={len(neg_s)} pos_ratio≈{len(pos_s)/len(combined):.3f}")

# ==========================================================================
# 4. Entry point
# ==========================================================================

def _parse_args():
    ap = argparse.ArgumentParser(
        description="alternative_sample_generation.py: gen-data (candidates -> "
        "LLM training csv) and mix (positive/negative blending).",
    )
    ap.add_argument(
        "--action", required=True, choices=["gen-data", "mix"],
        help="gen-data: build training csv from --phase state|mot. mix: blend --pos/--neg into --out.",
    )

    # --- gen-data flags ---
    ap.add_argument("--phase", type=str, default="state", choices=["state", "mot"])
    ap.add_argument(
        "--input_csv",
        type=str,
        default="./preds/transformer_llm_predictions_25u_loc_motivation_strict.csv",
        help="judge/llm_judge output CSV used to construct --phase state's alternative (counterfactual) examples.",
    )
    ap.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help=(
            "--phase state: override the alternative-data output path "
            "(default state_training_alternative_data.csv). "
            "--phase mot: override the output path (default motivation_training_data.csv)."
        ),
    )
    ap.add_argument(
        "--output_observed_csv",
        type=str,
        default=None,
        help="--phase state only: override the observed-data output path (default state_training_observed_data.csv).",
    )
    ap.add_argument(
        "--pkl-dir",
        type=Path,
        default=DG_PKL_DIR,
        help="PKL directory used to construct positive/true START rows (every PKL in it is used).",
    )
    ap.add_argument(
        "--location-map",
        type=Path,
        default=Path("../data/global_loc_map.pkl"),
        help="Location map used for subcategory IDs and true START rows.",
    )
    ap.add_argument(
        "--location-activity-map",
        type=Path,
        default=Path("../data/location_activity_map.pkl"),
        help="Subcategory-to-activity map used to classify true START rows.",
    )

    # --- mix flags ---
    ap.add_argument("--pos", type=str, default="state_training_observed_data.csv")
    ap.add_argument("--neg", type=str, default="state_training_alternative_data.csv")
    ap.add_argument("--out", type=str, default="state_training_data.csv")
    ap.add_argument("--pos_ratio", type=float, default=0.5, help="fraction of pos in final set (0.5 = 1:1 pos:neg)")
    ap.add_argument("--total", type=int, default=-1, help="final total rows; -1 means use all possible")
    ap.add_argument("--seed", type=int, default=3407)

    return ap.parse_args()


def main():
    args = _parse_args()
    if args.action == "gen-data":
        _run_gen_data(args)
    else:
        _run_mix(args)


if __name__ == "__main__":
    main()
