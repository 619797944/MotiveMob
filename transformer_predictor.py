from __future__ import annotations

import argparse
import math
import pickle
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

MODEL_DIR = Path("./predictors")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_TIME_BINS = 144
DAY_START_MIN = 4 * 60  # 04:00 -> 240 minutes


# ─────────────────────────────────────────────────────────
# Feature columns + model
# ─────────────────────────────────────────────────────────

CAT_COLS = [
    "cur_subcat_id", "cur_grid_id", "motivation_id",
    "weekday_start_loc", "weekday_end_loc",
    "weekend_start_loc", "weekend_end_loc",
    "freq_loc_1", "freq_loc_2", "freq_loc_3",
    "freq_loc_4", "freq_loc_5",
]
TIME_CAT_COLS = [
    "year", "is_weekend",
    "cur_time_bin",
    "weekday_start_time_bin", "weekday_end_time_bin",
    "weekend_start_time_bin", "weekend_end_time_bin",
    "freq_loc_1_time_bin", "freq_loc_2_time_bin",
    "freq_loc_3_time_bin", "freq_loc_4_time_bin",
    "freq_loc_5_time_bin",
]
USER_FAV_COLS = [
    "freq_id_1", "freq_id_2", "freq_id_3",
    "freq_id_4", "freq_id_5",
]

CAT_FAV_COLS = [
    "weekday_start_id", "weekday_end_id",
    "weekend_start_id", "weekend_end_id",
]

NUM_COLS = ["weekday_total_km", "weekend_total_km"]
POS_COLS = ["cur_lat", "cur_lng"]

# All categorical inputs fed into the model
ALL_CAT_COLS = CAT_COLS + TIME_CAT_COLS + USER_FAV_COLS
# All continuous inputs (merged into one numeric token)
ALL_NUM_COLS = NUM_COLS + POS_COLS


# ─────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────
class TrajDataset(Dataset):
    """
    cat_x : CAT_COLS + TIME_CAT_COLS + USER_FAV_COLS  [int64]
    num_x : NUM_COLS + POS_COLS                        [float32]
    """

    def __init__(self, df: pd.DataFrame):
        df = df.copy()
        self.cat_x  = torch.tensor(df[ALL_CAT_COLS].values, dtype=torch.int64)
        self.num_x  = torch.tensor(df[ALL_NUM_COLS].values, dtype=torch.float32)
        self.y_sub  = torch.tensor(df["next_subcat_id"].values, dtype=torch.int64)
        self.y_grid = torch.tensor(df["next_grid_id"].values,   dtype=torch.int64)
        self.y_time = torch.tensor(df["next_time_bin"].values,  dtype=torch.int64)

    def __len__(self):
        return len(self.y_sub)

    def __getitem__(self, idx):
        return (self.cat_x[idx], self.num_x[idx],
                self.y_sub[idx], self.y_grid[idx], self.y_time[idx])


# ─────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────
class ImprovedTabTransformer(nn.Module):
    """
    Token sequence:
      [CLS] | CAT_COLS tokens | TIME_CAT_COLS tokens | USER_FAV_COLS tokens | NUM token

    Each token = per-feature embedding + feature-group type embedding.
    The [CLS] output is used for all three classification heads.
    """

    N_CAT = len(CAT_COLS)
    N_TIME = len(TIME_CAT_COLS)
    N_FAV = len(USER_FAV_COLS)

    def __init__(
        self,
        cat_cards: list[int],       # cardinalities: CAT + TIME + USER_FAV (in order)
        num_numeric_feats: int,     # len(ALL_NUM_COLS)
        num_subcats: int,
        num_grids: int,
        num_timebins: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_ff: int = 256,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
    ):
        super().__init__()

        # Per-feature categorical embeddings
        self.cat_embs = nn.ModuleList([
            nn.Embedding(card, d_model) for card in cat_cards
        ])
        self.emb_drop = nn.Dropout(emb_dropout)

        # Numeric token projection (NUM_COLS + POS_COLS → d_model)
        self.num_proj = nn.Sequential(
            nn.Linear(num_numeric_feats, d_model),
            nn.LayerNorm(d_model),
        )

        # Feature-group type embeddings: 0=CAT_COLS, 1=TIME_CAT_COLS, 2=USER_FAV_COLS, 3=NUM token
        self.type_emb = nn.Embedding(4, d_model)

        # Build type_ids buffer: one per token position (no CLS)
        n_cat, n_time, n_fav = self.N_CAT, self.N_TIME, self.N_FAV
        type_ids = torch.cat([
            torch.zeros(n_cat,  dtype=torch.long),
            torch.ones(n_time,  dtype=torch.long),
            torch.full((n_fav,), 2, dtype=torch.long),
            torch.tensor([3],      dtype=torch.long),
        ])
        self.register_buffer("type_ids", type_ids)  # (n_cat+n_time+n_fav+1,)

        # [CLS] token — learnable
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Pre-LayerNorm Transformer encoder (norm_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # Pre-LN: more stable gradient flow
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Classification heads
        self.head_sub  = nn.Linear(d_model, num_subcats)
        self.head_grid = nn.Linear(d_model, num_grids)
        self.head_time = nn.Linear(d_model, num_timebins)

        self._init_weights()

    def forward(self, cat_x: torch.Tensor, num_x: torch.Tensor):
        """
        cat_x : (B, N_CAT + N_TIME + N_FAV)  int64
        num_x : (B, num_numeric_feats)        float32
        """
        B = cat_x.size(0)
        n_all_cat = self.N_CAT + self.N_TIME + self.N_FAV

        # Categorical tokens: (B, n_all_cat, D)
        cat_tokens = torch.stack(
            [self.cat_embs[i](cat_x[:, i]) for i in range(n_all_cat)], dim=1
        )

        # Numeric token: (B, 1, D)
        num_token = self.num_proj(num_x).unsqueeze(1)

        # Concat → (B, n_all_cat+1, D)
        seq = torch.cat([cat_tokens, num_token], dim=1)

        # Add feature-group type embeddings (broadcast over B)
        seq = seq + self.type_emb(self.type_ids).unsqueeze(0)
        seq = self.emb_drop(seq)

        # Prepend [CLS] → (B, 1 + n_all_cat+1, D)
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, seq], dim=1)

        enc = self.encoder(seq)   # (B, seq_len, D)
        pooled = enc[:, 0]        # [CLS] output  (B, D)

        return self.head_sub(pooled), self.head_grid(pooled), self.head_time(pooled)

    def _init_weights(self):
        for emb in self.cat_embs:
            nn.init.normal_(emb.weight, std=0.02)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        for head in (self.head_sub, self.head_grid, self.head_time):
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)


# ==========================================================================
# Reference data + pkl parsing + augmentation (alternative-motivation) rows
# (was alternative_sample_generation.py section 1, from transformer_judge.py)
# ==========================================================================

def invert_dict(d):
    return {value: key for key, value in d.items()}

PKL_DIR = Path("./seen_users")

GLOBAL_LOC_MAP_PATH = Path("./data/global_loc_map.pkl")

GRID_MAP_PATH = Path("./data/grid_map.pkl")

LOCATION_ACTIVITY_MAP_PATH = Path("./data/location_activity_map.pkl")

MODEL_CKPT_PATH = Path("./models") / "transformer_multihead_full_20.pt"

OUT_CSV_PATH = Path("transformer_predictions.csv")

OUT_PARQUET_PATH = Path("transformer_predictions.parquet")

JAPAN_LAT_MIN, JAPAN_LAT_MAX = 24.0, 46.5

JAPAN_LNG_MIN, JAPAN_LNG_MAX = 122.0, 153.0

loc_map = {}

map_loc = {}

subcat2id = {}

id2subcat = {}

grid2pois = {}

grid_id2key = {}

grid_key2id = {}

lat_min = None

lng_min = None

lat_step = None

lng_step = None

VALID_LOC_IDS = set()

activity_map = {}

START_CUR_SUBCAT_ID = None

START_CUR_GRID_ID = 0

START_CUR_BRANCH_ID = "0"

START_CUR_LAT = 0.0

START_CUR_LNG = 0.0

START_CUR_TIME = "04:00:00"

def load_reference_data(
    global_loc_map_path: Path,
    grid_map_path: Path,
    location_activity_map_path: Path = LOCATION_ACTIVITY_MAP_PATH,
):
    global loc_map, map_loc, subcat2id, id2subcat, activity_map
    global grid2pois, grid_id2key, grid_key2id
    global lat_min, lng_min, lat_step, lng_step, VALID_LOC_IDS, START_CUR_SUBCAT_ID

    with open(global_loc_map_path, "rb") as f:
        loc_map = pickle.load(f)

    map_loc = invert_dict(loc_map)

    loc_names = sorted(map_loc.keys())
    subcat_names = []
    for loc in loc_names:
        loc = loc.split("#")[0]
        if loc not in subcat_names:
            subcat_names.append(loc.strip())

    subcat2id = {name: idx for idx, name in enumerate(subcat_names)}
    id2subcat = {idx: name for name, idx in subcat2id.items()}
    START_CUR_SUBCAT_ID = len(subcat2id)

    with open(grid_map_path, "rb") as f:
        meta = pickle.load(f)

    grid2pois = meta["grid2pois"]
    grid_id2key = meta["grid_id2key"]
    grid_key2id = meta["grid_key2id"]

    lat_min = meta["lat_min"]
    lng_min = meta["lng_min"]
    lat_step = meta["lat_step"]
    lng_step = meta["lng_step"]
    VALID_LOC_IDS = set(loc_map.values())

    activity_path = location_activity_map_path
    if activity_path.exists():
        with open(activity_path, "rb") as f:
            activity_map = pickle.load(f)

MOTIVATION_LABELS = [
    "Return",
    "Personal Service",
    "Entertainment",
    "Dining",
    "Transit",
    "Religious",
    "Education",
    "Work",
    "Social",
    "Shopping",
    "Healthcare",
]

mot2id = {name: idx for idx, name in enumerate(MOTIVATION_LABELS)}

id2mot = invert_dict(mot2id)

all_mot_ids = list(mot2id.values())

def classify_motivation_from_top_subcat(top_category: Optional[str], subcategory: Optional[str]) -> Optional[str]:
    if top_category is None or subcategory is None:
        return None

    return_list = {
        "Hotel", "Bed and Breakfast", "Hostel", "Resort", "Inn",
        "Boarding House", "Lodging", "Motel", "Vacation Rental", "RV Park",
    }
    travel_personal_service = {
        "Fuel Station", "Electric Vehicle Charging Station",
        "Rental Car Location", "Bike Rental", "Boat Rental", "Baggage Locker",
        "Tourist Information and Service", "Travel Agency", "Tour Provider",
    }
    religious = {
        "Shrine", "Buddhist Temple", "Spiritual Center", "Church", "Mosque",
        "Temple", "Cemevi", "Confucian Temple", "Sikh Temple", "Prayer Room",
        "Hindu Temple", "Monastery", "Kingdom Hall", "Synagogue",
    }
    education = {
        "Music School", "University", "Community College", "Middle School",
        "Elementary School", "High School", "College and University",
        "College Academic Building", "Trade School", "Driving School", "Library",
        "College Engineering Building", "Preschool", "Adult Education", "Private School",
        "Religious School", "Medical School", "College Library", "College Classroom",
        "College Gym", "College Arts Building", "Primary and Secondary School",
        "College Quad", "College Bookstore", "College Lab", "Language School",
        "College Science Building", "College Auditorium", "Nursery School",
        "Culinary School", "College Theater", "College Stadium",
        "College Technology Building", "College Communications Building",
        "College Rec Center", "Summer Camp", "College Tennis Court", "Law School",
        "College Soccer Field", "College Baseball Diamond", "College Football Field",
        "College Track", "Circus School", "College Math Building", "Observatory",
        "Education", "Flight School", "Computer Training School",
    }
    return_home = {
        "Apartment or Condo", "College Residence Hall", "Home (private)",
        "Housing Development", "Residential Building", "Trailer Park", "Assisted Living",
    }
    work = {
        "Police Station", "Government Building", "City Hall", "Fire Station",
        "Courthouse", "Town Hall", "Military Base", "Capitol Building",
        "Embassy or Consulate", "Non-Profit Organization",
        "Social Services Organization", "Public and Social Service", "Labor Union",
        "Probation Office", "Polling Place", "Rescue Service", "Organization",
        "Prison", "College Administrative Building",
    }
    business_personal_service = {
        "Spa", "Massage Clinic", "Health and Beauty Service", "Nail Salon",
        "Hair Salon", "Bath House", "Repair Service", "Automotive Repair Shop",
        "Laundromat", "ATM", "Tire Repair Shop", "Barbershop", "Car Wash and Detail",
        "Photography Service", "Rental Service", "Pet Service", "Design Studio",
        "Dry Cleaner", "Insurance Agency", "Mover", "Laundry Service", "Credit Union",
        "Shoe Repair Service", "Machine Shop", "Tailor", "Child Care Service",
        "Tattoo Parlor", "Skin Care Clinic", "Locksmith", "Tanning Salon",
        "Body Piercing Shop", "Notary", "Check Cashing Service",
        "Computer Repair Service", "Automotive Service", "Motorcycle Repair Shop",
        "Daycare", "Financial Service", "Bank", "Lottery Retailer", "Photography Lab",
        "Waste Management Service", "Home Service", "Event Service", "Storage Facility",
        "Shipping, Freight, and Material Transportation Service", "Employment Agency",
        "Vehicle Inspection Station", "IT Service", "Currency Exchange",
        "Water Treatment Service", "Agriculture and Forestry Service",
        "Telecommunication Service", "Real Estate Service", "Import and Export Service",
        "Professional Cleaning Service", "Equipment Rental Service", "Painter",
        "Legal Service", "Tutoring Service",
    }

    if top_category == "Travel and Transportation":
        if subcategory in return_list:
            return "Return"
        if subcategory in travel_personal_service:
            return "Personal Service"
        if subcategory in {"Cruise", "Hotel Pool"}:
            return "Entertainment"
        if subcategory == "Airport Food Court":
            return "Dining"
        return "Transit"
    if top_category == "Community and Government":
        if subcategory in religious:
            return "Religious"
        if subcategory in education:
            return "Education"
        if subcategory in return_home:
            return "Return"
        if subcategory in work:
            return "Work"
        if subcategory in {"Post Office", "Public Bathroom", "Animal Shelter", "Rehabilitation Center"}:
            return "Personal Service"
        if subcategory == "College Cafeteria":
            return "Dining"
        return "Social"
    if top_category == "Business and Professional Services":
        if subcategory in business_personal_service:
            return "Personal Service"
        if subcategory in {"Wedding Hall", "Funeral Home"}:
            return "Social"
        if subcategory in {
            "Auditorium", "Ballroom",
            "Print, TV, Radio and Outdoor Advertising Service",
            "Entertainment Service",
        }:
            return "Entertainment"
        if subcategory == "Food and Beverage Service":
            return "Shopping"
        return "Work"
    if top_category == "Dining and Drinking":
        return "Dining"
    if top_category == "Arts and Entertainment":
        return "Entertainment"
    if top_category == "Retail":
        return "Shopping"
    if top_category == "Health and Medicine":
        return "Healthcare"
    if top_category == "Landmarks and Outdoors":
        return "Entertainment"
    if top_category == "Sports and Recreation":
        return "Entertainment"
    if top_category == "Event":
        return "Social"
    return None

def motivation_for_first_step(first_step: str) -> int:
    first_loc_id, _first_time = first_step.split(" at ")
    first_subcat = first_loc_id.strip().split("#")[0].strip()
    top_category = activity_map.get(first_subcat)
    motivation = classify_motivation_from_top_subcat(top_category, first_subcat)
    if motivation not in mot2id:
        raise ValueError(
            f"Cannot classify START motivation for first POI subcategory "
            f"{first_subcat!r} with top category {top_category!r}"
        )
    return mot2id[motivation]

def latlng_to_grid(lat, lng, lat_min, lng_min, lat_step, lng_step):
    gx = int((lat - lat_min) / lat_step)
    gy = int((lng - lng_min) / lng_step)
    return gx, gy

def get_lat_lng(loc_name):
    pos = loc_name.split("(")[-1].split(")")[0]

    lat_str, lng_str = [x.strip() for x in pos.split(",")]
    lat = float(lat_str)
    lng = float(lng_str)

    return lat, lng

def parse_user_pkl(obj: Any) -> Tuple[List[str], Any, str]:
    # returns (trajectories, motivations, behavior_report)
    # Current format is [trajs, test, motos, report, report_with_covid_context];
    if isinstance(obj, list) and len(obj) == 5:
        trajs, test, motos, report, _report_covid = obj
        return trajs, motos, report
    if isinstance(obj, list) and len(obj) == 4:
        trajs, test, motos, report = obj
        return trajs, motos, report

    raise ValueError("Unsupported pkl structure; expected dict or (trajs, motos, report).")

def extract_date_and_steps(traj: str) -> Tuple[Optional[str], List[str]]:
    # "Activities at 2019-01-05: A at 22:40:00, B at 01:30:00"
    # returns ("2019-01-05", ["A at 22:40:00", "B at 01:30:00"])
    s = " ".join(traj.split())
    date = None

    try:
        prefix = s.split(":", 1)[0]
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

def time_str_to_bin(t_str: str) -> int:
    """
    t_str: 'HH:MM:SS'
    Returns a time-bin index in 0~143.
    Convention: bin 0 = 04:00:00, bin 143 = 03:50:00 the next day.
    """
    h, m, s = map(int, t_str.split(":"))
    total_min = h * 60 + m

    shifted = (total_min - DAY_START_MIN) % (24 * 60)
    bin_idx = shifted // 10

    return int(bin_idx)

def bin_to_time_str(bin_idx: int) -> str:
    """
    0 -> '04:00:00'
    1 -> '04:10:00'
    ...
    143 -> '03:50:00' (early morning of the next day)
    """
    if not (0 <= bin_idx < NUM_TIME_BINS):
        raise ValueError(
            f"bin_idx should be in [0, {NUM_TIME_BINS - 1}], got {bin_idx}"
        )

    shifted_min = bin_idx * 10
    total_min = (shifted_min + DAY_START_MIN) % (24 * 60)

    h = total_min // 60
    m = total_min % 60

    return f"{h:02d}:{m:02d}:00"

def is_weekend(date):
    return int(date.weekday() >= 5)

def split_checkin_step(step: str) -> Tuple[str, str, str, str]:
    loc_id, time_str = step.split(" at ")
    subcat, branch_id = (
        loc_id.strip().split("#")[0],
        loc_id.strip().split("#")[1],
    )
    return loc_id.strip(), time_str.strip(), subcat.strip(), branch_id.strip()

def get_grid_id_for_loc_id(loc_id: str) -> Tuple[float, float, int]:
    lat, lng = get_lat_lng(map_loc[loc_id])
    if (JAPAN_LAT_MIN <= lat <= JAPAN_LAT_MAX) and (
        JAPAN_LNG_MIN <= lng <= JAPAN_LNG_MAX
    ):
        gx, gy = latlng_to_grid(
            lat,
            lng,
            lat_min,
            lng_min,
            lat_step,
            lng_step,
        )
        grid_id = grid_key2id[(gx, gy)]
    else:
        grid_id = 0
    return lat, lng, grid_id

def build_feature_transition_row(
    *,
    person: Path,
    date,
    year: int,
    month: int,
    day: int,
    is_weekend_flag: int,
    feats: Dict[str, Any],
    original: str,
    next_step: str,
    motivation_id: int,
) -> Dict[str, Any]:
    next_loc_id, next_time, next_subcat, next_branch_id = split_checkin_step(next_step)
    next_subcat_id = subcat2id[next_subcat]
    _next_lat, _next_lng, next_grid_id = get_grid_id_for_loc_id(next_loc_id)

    if str(original).strip() == "START":
        cur_time = START_CUR_TIME
        cur_time_bin = time_str_to_bin(cur_time)
        cur_subcat_id = START_CUR_SUBCAT_ID
        cur_grid_id = START_CUR_GRID_ID
        cur_lat = START_CUR_LAT
        cur_lng = START_CUR_LNG
        cur_branch_id = START_CUR_BRANCH_ID
        cur_loc_name = START_CUR_SUBCAT_ID
    else:
        cur_loc_id, cur_time, cur_subcat, cur_branch_id = split_checkin_step(original)
        cur_time_bin = time_str_to_bin(cur_time)
        cur_subcat_id = subcat2id[cur_subcat]
        cur_lat, cur_lng, cur_grid_id = get_grid_id_for_loc_id(cur_loc_id)
        cur_loc_name = cur_subcat_id

    row = {
        "user": Path(person).stem,
        "person": person,
        "current_checkin": original,
        "next_checkin": next_step,
        "original": original,
        "date": date,
        "year": year,
        "month": month,
        "day": day,
        "is_weekend": is_weekend_flag,
        "cur_loc_name": cur_loc_name,
        "cur_time": cur_time,
        "cur_subcat_id": cur_subcat_id,
        "cur_grid_id": cur_grid_id,
        "cur_time_bin": cur_time_bin,
        "cur_lat": cur_lat,
        "cur_lng": cur_lng,
        "cur_id": cur_branch_id,
        "next_subcat_id": next_subcat_id,
        "next_grid_id": next_grid_id,
        "next_time_bin": time_str_to_bin(next_time),
        "next_id": next_branch_id,
        "motivation_id": motivation_id,
    }
    row.update(feats)
    return row

def parse_report(report):
    """
    Input: one report string.
    Output: a dict of structured features.
    """
    feat = {}

    # --------- weekday section ---------
    m = re.search(r"During weekday, you usually travel over (\d+) kilometers", report)
    feat["weekday_total_km"] = int(m.group(1)) if m else 0

    m = re.search(r"begin your daily trip at (\d{2}:\d{2}:\d{2})", report)
    feat["weekday_start_time_bin"] = time_str_to_bin(m.group(1)) if m else -1

    m = re.search(r"end your daily trip at (\d{2}:\d{2}:\d{2})", report)
    feat["weekday_end_time_bin"] = time_str_to_bin(m.group(1)) if m else -1

    m = re.search(r"weekday.*?visit (.+?#\d+) at the beginning", report)
    loc, id = m.group(1).strip().split("#")[0], m.group(1).strip().split("#")[1]
    feat["weekday_start_loc"] = subcat2id.get(loc, -1) if loc else -1
    feat["weekday_start_id"] = id

    m = re.search(r"go to (.+?#\d+) before returning home", report)
    loc, id = m.group(1).strip().split("#")[0], m.group(1).strip().split("#")[1]
    feat["weekday_end_loc"] = subcat2id.get(loc, -1) if loc else -1
    feat["weekday_end_id"] = id

    # --------- weekend section ---------
    m = re.search(r"During weekend, you usually travel over (\d+) kilometers", report)
    feat["weekend_total_km"] = int(m.group(1)) if m else 0

    m = re.findall(r"weekend.*?begin your daily trip at (\d{2}:\d{2}:\d{2})", report)
    feat["weekend_start_time_bin"] = time_str_to_bin(m[0]) if m else -1

    m = re.findall(r"weekend.*?end your daily trip at (\d{2}:\d{2}:\d{2})", report)
    feat["weekend_end_time_bin"] = time_str_to_bin(m[0]) if m else -1

    m = re.search(r"weekend.*?visit (.+?#\d+) at the beginning", report)
    loc, id = m.group(1).strip().split("#")[0], m.group(1).strip().split("#")[1]
    feat["weekend_start_loc"] = subcat2id.get(loc, -1) if loc else -1
    feat["weekend_start_id"] = id

    m = re.search(r"weekend.*?go to (.+?#\d+) before returning home", report)
    loc, id = m.group(1).strip().split("#")[0], m.group(1).strip().split("#")[1]
    feat["weekend_end_loc"] = subcat2id.get(loc, -1) if loc else -1
    feat["weekend_end_id"] = id

    # -------- Frequent visits --------
    m = re.search(r"You usually visit (.+)", report)

    buffer = ""
    steps = []

    for p in m.group(1).split(","):
        if "00" in p.strip():
            p = buffer + p.strip()
            loc, time = p.strip().split(" at ")
            steps.append((loc, time))
            buffer = ""
        else:
            buffer = buffer + p.strip() + ", "

    for i in range(5):
        if i < len(steps):
            loc_id, t = steps[i]
            loc, id = loc_id.strip().split("#")[0], loc_id.strip().split("#")[1]

            feat[f"freq_loc_{i + 1}"] = subcat2id.get(loc, -1) if loc else -1
            feat[f"freq_id_{i + 1}"] = id
            feat[f"freq_loc_{i + 1}_time_bin"] = time_str_to_bin(t)
        else:
            feat[f"freq_loc_{i + 1}"] = -1
            feat[f"freq_id_{i + 1}"] = -1
            feat[f"freq_loc_{i + 1}_time_bin"] = -1

    return feat

def construct_df(
    pkl_dir: Path = PKL_DIR,
    alternatives_per_transition: int = 3,
    include_start_transition: bool = True,
):
    """
    Build alternative-motivation (augmentation) transition rows.

    Uses every .pkl directly under pkl_dir (e.g. seen_users) - no
    sample-file whitelist / seed-pkl / sample-limit subsetting.
    """
    if alternatives_per_transition < 1:
        raise ValueError("alternatives_per_transition must be at least 1")
    if alternatives_per_transition >= len(all_mot_ids):
        raise ValueError(
            "alternatives_per_transition must be smaller than the number of motivation labels"
        )

    sample_set = sorted(pkl_dir.rglob("*.pkl"))

    print(sample_set)
    print("Found pkls:", len(sample_set))

    rows = []

    for p in sample_set:
        with open(p, "rb") as f:
            obj = pickle.load(f)

        trajectories, motivations, report = parse_user_pkl(obj)

        for ti, traj in enumerate(trajectories):
            date, steps = extract_date_and_steps(traj)

            if len(steps) < 1:
                continue

            date = pd.to_datetime(date)
            year = date.year
            month = date.month
            day = date.day
            weekend_flag = is_weekend(date)
            feats = parse_report(report)

            if include_start_transition:
                start_motivation = motivation_for_first_step(steps[0])
                candidates = [m for m in all_mot_ids if m != start_motivation]
                neg_mots = random.sample(
                    candidates,
                    k=alternatives_per_transition,
                )
                for mot in neg_mots:
                    rows.append(
                        build_feature_transition_row(
                            person=p,
                            date=date,
                            year=year,
                            month=month,
                            day=day,
                            is_weekend_flag=weekend_flag,
                            feats=feats,
                            original="START",
                            next_step=steps[0],
                            motivation_id=mot,
                        )
                    )

            for i in range(len(steps) - 1):
                motivation = mot2id[motivations[ti][i]]

                candidates = [m for m in all_mot_ids if m != motivation]
                neg_mots = random.sample(
                    candidates,
                    k=alternatives_per_transition,
                )

                for mot in neg_mots:
                    rows.append(
                        build_feature_transition_row(
                            person=p,
                            date=date,
                            year=year,
                            month=month,
                            day=day,
                            is_weekend_flag=weekend_flag,
                            feats=feats,
                            original=steps[i],
                            next_step=steps[i + 1],
                            motivation_id=mot,
                        )
                    )

    df = pd.DataFrame(rows)

    print("Total transitions:", len(df))
    df = df.dropna(how="any")
    print("Total transitions after drop:", len(df))

    df["date"] = pd.to_datetime(df["date"])

    return df


# ==========================================================================
# Train mode: build training data directly from pkls, train ImprovedTabTransformer
# (was transformer_training.py / transformer_predictor.py)
# ==========================================================================

def build_and_split_data(
    pkl_dir: Path,
    valid_frac: float = 0.2,
    seed: int = 42,
):
    """
    Build positive (true-motivation) transition rows directly from the user
    .pkl files under pkl_dir - no intermediate parquet file. Split into
    train/valid by day (not by row) so a user's history for one day doesn't
    leak across the split.

    Call load_reference_data() first to populate map_loc/subcat2id/grid state.
    """
    pkl_files = sorted(Path(pkl_dir).rglob("*.pkl"))
    print("Found pkls:", len(pkl_files))

    rows = []
    for p in pkl_files:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        trajectories, motivations, report = parse_user_pkl(obj)
        feats = parse_report(report)

        for ti, traj in enumerate(trajectories):
            date, steps = extract_date_and_steps(traj)
            if len(steps) < 2:
                continue
            date = pd.to_datetime(date)
            weekend_flag = is_weekend(date)

            for i in range(len(steps) - 1):
                cur_loc_id, cur_time = steps[i].split(" at ")
                next_loc_id, next_time = steps[i + 1].split(" at ")
                cur_loc_id, next_loc_id = cur_loc_id.strip(), next_loc_id.strip()

                cur_loc, _cur_id = cur_loc_id.split("#")[0].strip(), cur_loc_id.split("#")[1]
                next_loc, next_id = next_loc_id.split("#")[0].strip(), next_loc_id.split("#")[1]
                cur_lat, cur_lng, cur_grid_id = get_grid_id_for_loc_id(cur_loc_id)
                next_lat, next_lng, next_grid_id = get_grid_id_for_loc_id(next_loc_id)

                row = {
                    "person": str(p),
                    "date": date,
                    "year": date.year,
                    "month": date.month,
                    "day": date.day,
                    "is_weekend": weekend_flag,
                    "cur_time": cur_time,
                    "cur_subcat_id": subcat2id[cur_loc],
                    "cur_grid_id": cur_grid_id,
                    "cur_time_bin": time_str_to_bin(cur_time),
                    "cur_lat": cur_lat,
                    "cur_lng": cur_lng,
                    "next_subcat_id": subcat2id[next_loc],
                    "next_grid_id": next_grid_id,
                    "next_time_bin": time_str_to_bin(next_time),
                    "next_id": next_id,
                    "motivation_id": mot2id[motivations[ti][i]],
                }
                row.update(feats)
                rows.append(row)

    df = pd.DataFrame(rows)
    print("Total transitions:", len(df))
    df = df.dropna(how="any")
    print("Total transitions after drop:", len(df))
    df["date"] = pd.to_datetime(df["date"])

    unique_dates = df["date"].dt.date.unique()
    rng = np.random.RandomState(seed)
    n_valid_dates = max(1, int(valid_frac * len(unique_dates)))
    valid_dates = rng.choice(unique_dates, size=n_valid_dates, replace=False)
    valid_mask = df["date"].dt.date.isin(valid_dates)
    train_df = df[~valid_mask].copy()
    valid_df = df[valid_mask].copy()

    print(f"Total dates: {len(unique_dates)}  Valid dates: {len(valid_dates)}")
    print(f"Train: {len(train_df):,}  Valid: {len(valid_df):,}")
    return train_df, valid_df


def preprocess(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    """Cast types, drop NaN labels, return cardinalities."""
    for c in ALL_CAT_COLS:
        for df in (train_df, valid_df):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")

    for c in ALL_NUM_COLS:
        for df in (train_df, valid_df):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")

    for label in ["next_subcat_id", "next_grid_id", "next_time_bin", "next_id"]:
        train_df[label] = pd.to_numeric(train_df[label], errors="coerce")
        valid_df[label]  = pd.to_numeric(valid_df[label],  errors="coerce")
        train_df = train_df.dropna(subset=[label])
        valid_df  = valid_df.dropna(subset=[label])
        train_df[label] = train_df[label].astype("int64")
        valid_df[label]  = valid_df[label].astype("int64")

    cat_cards = []
    for c in ALL_CAT_COLS:
        mx = int(max(train_df[c].max(), valid_df[c].max()))
        cat_cards.append(max(mx + 1, 2))

    num_subcats  = int(max(train_df["next_subcat_id"].max(), valid_df["next_subcat_id"].max()) + 1)
    num_grids    = int(max(train_df["next_grid_id"].max(),   valid_df["next_grid_id"].max())   + 1)
    num_timebins = int(max(train_df["next_time_bin"].max(),  valid_df["next_time_bin"].max())  + 1)

    return train_df, valid_df, cat_cards, num_subcats, num_grids, num_timebins


def _cosine_warmup_lambda(warmup_steps: int, total_steps: int, min_ratio: float = 0.05):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


@torch.no_grad()
def _topk_hits(logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    _, topk = torch.topk(logits, k=min(k, logits.size(1)), dim=1)
    return (topk == labels.unsqueeze(1)).any(dim=1).sum().item()


def train_and_validate(
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 3,
    dim_ff: int = 256,
    dropout: float = 0.1,
    emb_dropout: float = 0.1,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 2048,
    epochs: int = 30,
    warmup_epochs: int = 2,
    grad_clip: float = 1.0,
    label_smoothing: float = 0.05,
    w_sub: float = 1 / 3,
    w_grid: float = 1 / 3,
    w_time: float = 1 / 3,
    topk: tuple = (1, 3, 5),
    pkl_dir: str | Path | None = None,
    global_loc_map: str | Path | None = None,
    grid_map: str | Path | None = None,
    location_activity_map: str | Path | None = None,
    valid_frac: float = 0.2,
    seed: int = 42,
    model_dir: str | Path | None = None,
    save_path: str | Path | None = None,
) -> float:
    """
    Returns best composite validation metric (mean top-1 accuracy across 3 heads).
    """
    out_dir = Path(model_dir) if model_dir else MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_path is None:
        save_path = out_dir / "transformer_improved_best.pt"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize loss weights
    tw = w_sub + w_grid + w_time
    w_sub, w_grid, w_time = w_sub / tw, w_grid / tw, w_time / tw

    # ── Data ──
    load_reference_data(
        global_loc_map_path=Path(global_loc_map) if global_loc_map else GLOBAL_LOC_MAP_PATH,
        grid_map_path=Path(grid_map) if grid_map else GRID_MAP_PATH,
        location_activity_map_path=Path(location_activity_map) if location_activity_map else LOCATION_ACTIVITY_MAP_PATH,
    )
    train_df, valid_df = build_and_split_data(
        pkl_dir=Path(pkl_dir) if pkl_dir else PKL_DIR,
        valid_frac=valid_frac,
        seed=seed,
    )
    train_df, valid_df, cat_cards, num_subcats, num_grids, num_timebins = preprocess(
        train_df, valid_df
    )
    n_num = len(ALL_NUM_COLS)

    print(
        f"num_subcats={num_subcats}, num_grids={num_grids}, num_timebins={num_timebins}\n"
        f"cat_tokens={len(ALL_CAT_COLS)}, numeric_feats={n_num}"
    )

    train_ds = TrajDataset(train_df)
    valid_ds = TrajDataset(valid_df)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # ── Model ──
    model = ImprovedTabTransformer(
        cat_cards=cat_cards,
        num_numeric_feats=n_num,
        num_subcats=num_subcats,
        num_grids=num_grids,
        num_timebins=num_timebins,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_ff=dim_ff,
        dropout=dropout,
        emb_dropout=emb_dropout,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    crit = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    steps_per_epoch = math.ceil(len(train_ds) / batch_size)
    total_steps     = epochs * steps_per_epoch
    warmup_steps    = warmup_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _cosine_warmup_lambda(warmup_steps, total_steps)
    )

    best_metric = -1.0

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        tr_loss, tr_n = 0.0, 0

        for cat_x, num_x, y_sub, y_grid, y_time in train_loader:
            cat_x  = cat_x.to(DEVICE)
            num_x  = num_x.to(DEVICE)
            y_sub  = y_sub.to(DEVICE)
            y_grid = y_grid.to(DEVICE)
            y_time = y_time.to(DEVICE)

            optimizer.zero_grad()
            l_sub, l_grid, l_time = model(cat_x, num_x)
            loss = (
                w_sub  * crit(l_sub,  y_sub)
                + w_grid * crit(l_grid, y_grid)
                + w_time * crit(l_time, y_time)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            bs = y_sub.size(0)
            tr_loss += loss.item() * bs
            tr_n    += bs

        # ── Validate ──
        model.eval()
        vl_loss, vl_n = 0.0, 0
        hits = {h: {k: 0 for k in topk} for h in ("sub", "grid", "time")}

        with torch.no_grad():
            for cat_x, num_x, y_sub, y_grid, y_time in valid_loader:
                cat_x  = cat_x.to(DEVICE)
                num_x  = num_x.to(DEVICE)
                y_sub  = y_sub.to(DEVICE)
                y_grid = y_grid.to(DEVICE)
                y_time = y_time.to(DEVICE)

                l_sub, l_grid, l_time = model(cat_x, num_x)
                loss = (
                    w_sub  * crit(l_sub,  y_sub)
                    + w_grid * crit(l_grid, y_grid)
                    + w_time * crit(l_time, y_time)
                )
                bs = y_sub.size(0)
                vl_loss += loss.item() * bs
                vl_n    += bs

                for k in topk:
                    hits["sub"][k]  += _topk_hits(l_sub,  y_sub,  k)
                    hits["grid"][k] += _topk_hits(l_grid, y_grid, k)
                    hits["time"][k] += _topk_hits(l_time, y_time, k)

        accs = {h: {k: hits[h][k] / vl_n for k in topk} for h in hits}
        metric = (accs["sub"][1] + accs["grid"][1] + accs["time"][1]) / 3
        cur_lr = scheduler.get_last_lr()[0]

        print(
            f"[Ep {epoch:03d}] lr={cur_lr:.2e} "
            f"tr={tr_loss/tr_n:.4f} vl={vl_loss/vl_n:.4f} | "
            f"sub@(1,3,5)=({accs['sub'][1]:.4f},{accs['sub'][3]:.4f},{accs['sub'][5]:.4f}) "
            f"grid@(1,3,5)=({accs['grid'][1]:.4f},{accs['grid'][3]:.4f},{accs['grid'][5]:.4f}) "
            f"time@(1,3,5)=({accs['time'][1]:.4f},{accs['time'][3]:.4f},{accs['time'][5]:.4f})"
        )

        if metric > best_metric:
            best_metric = metric
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "cat_cards": cat_cards,
                    "num_numeric_feats": n_num,
                    "num_subcats": num_subcats,
                    "num_grids": num_grids,
                    "num_timebins": num_timebins,
                    "hparams": dict(
                        d_model=d_model, nhead=nhead, num_layers=num_layers,
                        dim_ff=dim_ff, dropout=dropout, emb_dropout=emb_dropout,
                    ),
                    "loss_weights": dict(w_sub=w_sub, w_grid=w_grid, w_time=w_time),
                    "best_epoch": epoch,
                    "val_metric": best_metric,
                    "topk_accs": {k: {h: accs[h][k] for h in accs} for k in topk},
                },
                save_path,
            )
            print(f"  *** New best (metric={best_metric:.4f}) → {save_path}")

    print(f"Done. Best composite top-1 metric = {best_metric:.4f}")
    del model
    torch.cuda.empty_cache()
    return best_metric

# ==========================================================================
# Predict mode: candidate generation + rule-based pick
# ==========================================================================

def softmax_from_log_dict(log_dict, top_k=None):
    """
    log_dict: {key -> log_score}
    Returns sorted keys and normalized probabilities.
    """
    if not log_dict:
        return [], []

    keys = list(log_dict.keys())
    logs = np.array([log_dict[k] for k in keys], dtype=np.float64)

    m = logs.max()
    exps = np.exp(logs - m)
    probs = exps / exps.sum()

    order = np.argsort(-probs)
    if top_k is not None:
        order = order[:top_k]

    keys_sorted = [keys[i] for i in order]
    probs_sorted = probs[order]
    return keys_sorted, probs_sorted

def collect_transformer_candidates(
    valid_df,
    grid_map_path: Path = GRID_MAP_PATH,
    model_ckpt_path: Path = MODEL_CKPT_PATH,
    K_sub=5,
    K_grid=5,
    K_time=5,
    K_pair_out=10,
    K_triple_out=10,
    alpha_sub=0.8,
    alpha_grid=0.9,
    alpha_time=0.55,
    alpha_user=1.0,
    lambda_d=0.0,
):
    """
    Generate Transformer pair/triple candidates locally.

    pair_cands[i]   = [((subcat_id, branch_id), prob), ...]
    triple_cands[i] = [((subcat_id, branch_id, time_bin), prob), ...]
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(
        model_ckpt_path,
        map_location=device,
    )

    with open(grid_map_path, "rb") as f:
        meta = pickle.load(f)
    grid2subcat2pois = meta["grid2pois"]

    if "cat_cards" not in ckpt:
        raise ValueError(
            f"{model_ckpt_path} is not an ImprovedTabTransformer checkpoint "
            "(missing 'cat_cards'); train one with transformer_training.py."
        )
    valid_dataset = TrajDataset(valid_df)
    model = ImprovedTabTransformer(
        cat_cards=ckpt["cat_cards"],
        num_numeric_feats=ckpt["num_numeric_feats"],
        num_subcats=ckpt["num_subcats"],
        num_grids=ckpt["num_grids"],
        num_timebins=ckpt["num_timebins"],
        **ckpt["hparams"],
    ).to(device)

    valid_loader = DataLoader(valid_dataset, batch_size=2048, shuffle=False, num_workers=0)
    N = len(valid_dataset)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded transformer judge model: ImprovedTabTransformer from {model_ckpt_path}")

    pair_cands = [[] for _ in range(N)]
    triple_cands = [[] for _ in range(N)]

    eps = 1e-9
    global_i = 0

    with torch.no_grad():
        for batch in valid_loader:
            cat_x, num_x, y_subcat, y_grid, y_time = batch

            cat_x = cat_x.to(device)
            num_x = num_x.to(device)

            logits_subcat, logits_grid, logits_time = model(cat_x, num_x)

            prob_sub = F.softmax(logits_subcat, dim=1)
            prob_grid = F.softmax(logits_grid, dim=1)
            prob_time = F.softmax(logits_time, dim=1)

            sub_probs_topk, sub_idx_topk = torch.topk(prob_sub, k=K_sub, dim=1)
            grd_probs_topk, grd_idx_topk = torch.topk(prob_grid, k=K_grid, dim=1)
            time_probs_topk, time_idx_topk = torch.topk(prob_time, k=K_time, dim=1)

            bs = y_subcat.size(0)

            for bi in range(bs):
                idx = global_i + bi
                row = valid_df.iloc[idx]
                is_start_row = str(row.get("original", "")).strip() == "START"

                lat0 = None if is_start_row else float(row["cur_lat"])
                lng0 = None if is_start_row else float(row["cur_lng"])

                is_we = int(row["is_weekend"])
                cur_tb = int(row["cur_time_bin"])

                if is_we == 0:
                    u_start_id = int(row["weekday_start_id"])
                    u_end_id = int(row["weekday_end_id"])
                    u_start_tb = int(row["weekday_start_time_bin"])
                    u_end_tb = int(row["weekday_end_time_bin"])
                else:
                    u_start_id = int(row["weekend_start_id"])
                    u_end_id = int(row["weekend_end_id"])
                    u_start_tb = int(row["weekend_start_time_bin"])
                    u_end_tb = int(row["weekend_end_time_bin"])

                use_start = (u_start_tb >= 0) and (abs(cur_tb - u_start_tb) <= 2)
                use_end = (u_end_tb >= 0) and (abs(cur_tb - u_end_tb) <= 2)

                fav_list = [
                    row.get(f"freq_id_{j}", 0)
                    for j in range(1, 6)
                ]
                fav_set = set(int(b) for b in fav_list if int(b) > 0)

                candidate_triple_log = {}
                candidate_pair_log = {}

                for s_id_tensor, p_s in zip(sub_idx_topk[bi], sub_probs_topk[bi]):
                    s_id = int(s_id_tensor.item())
                    p_s_val = float(p_s.item())
                    if p_s_val <= 0:
                        continue
                    log_ps = math.log(p_s_val + eps)

                    for g_id_tensor, p_g in zip(grd_idx_topk[bi], grd_probs_topk[bi]):
                        g_id = int(g_id_tensor.item())
                        p_g_val = float(p_g.item())
                        if p_g_val <= 0:
                            continue
                        log_pg = math.log(p_g_val + eps)

                        subcat_dict = grid2subcat2pois.get(g_id, None)
                        if subcat_dict is None:
                            continue

                        poi_list = subcat_dict.get(s_id, [])
                        if not poi_list:
                            continue

                        for t_id_tensor, p_t in zip(time_idx_topk[bi], time_probs_topk[bi]):
                            t_id = int(t_id_tensor.item())
                            p_t_val = float(p_t.item())
                            if p_t_val <= 0:
                                continue
                            log_pt = math.log(p_t_val + eps)

                            for poi in poi_list:
                                b_id = int(poi["branch_id"])
                                lat_p = float(poi["lat"])
                                lng_p = float(poi["lng"])

                                if lat0 is None or lng0 is None:
                                    d2 = 0.0
                                else:
                                    d2 = (lat_p - lat0) ** 2 + (lng_p - lng0) ** 2

                                user_bonus = 1.0 if b_id in fav_set else 0.0
                                if use_start and b_id == u_start_id:
                                    user_bonus = 1.5
                                if use_end and b_id == u_end_id:
                                    user_bonus = 1.5

                                log_score = (
                                    alpha_sub * log_ps
                                    + alpha_grid * log_pg
                                    + alpha_time * log_pt
                                    - lambda_d * d2
                                    + alpha_user * user_bonus
                                )

                                triple_key = (s_id, b_id, t_id)
                                if (
                                    triple_key not in candidate_triple_log
                                    or log_score > candidate_triple_log[triple_key]
                                ):
                                    candidate_triple_log[triple_key] = log_score

                                pair_key = (s_id, b_id)
                                if (
                                    pair_key not in candidate_pair_log
                                    or log_score > candidate_pair_log[pair_key]
                                ):
                                    candidate_pair_log[pair_key] = log_score

                if candidate_triple_log:
                    tri_keys, tri_probs = softmax_from_log_dict(
                        candidate_triple_log,
                        top_k=K_triple_out,
                    )
                    triple_cands[idx] = list(zip(tri_keys, tri_probs))

                    pair_keys, pair_probs = softmax_from_log_dict(
                        candidate_pair_log,
                        top_k=K_pair_out,
                    )
                    pair_cands[idx] = list(zip(pair_keys, pair_probs))

            global_i += bs

    return pair_cands, triple_cands


def sanity_check_transformer_output(
    subcat_id: Any,
    branch_id: Any,
    time_bin: Any,
    cur_time_bin: int,
    allow_equal_time: bool = False,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Checks whether Transformer prediction is usable.

    Constraints:
      1) time_bin in [0, 143]
      2) time_bin > cur_time_bin, except START may allow equality
      3) subcategory_name#branch_id exists in loc_map.values()

    Returns:
      (ok, err_msg, text_out_if_ok)
    """
    try:
        subcat_id = int(subcat_id)
        branch_id = int(branch_id)
        time_bin = int(time_bin)
    except Exception:
        return False, "invalid_type: subcat_id/branch_id/time_bin not int-castable", None

    if not (0 <= time_bin < NUM_TIME_BINS):
        return False, f"time_bin_out_of_range: {time_bin}", None

    if int(cur_time_bin) < 0:
        return False, f"invalid_cur_time_bin: {cur_time_bin}", None

    if allow_equal_time:
        time_is_invalid = time_bin < int(cur_time_bin)
    else:
        time_is_invalid = time_bin <= int(cur_time_bin)

    if time_is_invalid:
        return False, f"time_not_after_current: pred={time_bin} cur={cur_time_bin}", None

    subcat_name = id2subcat.get(subcat_id, f"subcat_{subcat_id}")
    loc_str = f"{subcat_name}#{branch_id}"
    time_str = bin_to_time_str(time_bin)
    text_out = f"{loc_str} at {time_str}"

    if loc_str not in VALID_LOC_IDS:
        return False, f"loc_not_in_loc_map_values: {loc_str}", None

    return True, None, text_out


def _parse_args_train():
    parser = argparse.ArgumentParser(description="Train ImprovedTabTransformer")
    parser.add_argument("--d_model",         type=int,   default=128)
    parser.add_argument("--nhead",           type=int,   default=4)
    parser.add_argument("--num_layers",      type=int,   default=3)
    parser.add_argument("--dim_ff",          type=int,   default=256)
    parser.add_argument("--dropout",         type=float, default=0.1)
    parser.add_argument("--emb_dropout",     type=float, default=0.1)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--weight_decay",    type=float, default=1e-4)
    parser.add_argument("--batch_size",      type=int,   default=2048)
    parser.add_argument("--epochs",          type=int,   default=30)
    parser.add_argument("--warmup_epochs",   type=int,   default=2)
    parser.add_argument("--grad_clip",       type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--w_sub",           type=float, default=1/3)
    parser.add_argument("--w_grid",          type=float, default=1/3)
    parser.add_argument("--w_time",          type=float, default=1/3)
    parser.add_argument("--pkl_dir",         type=str,   default=None,
                        help="directory of user .pkl files (default: ./seen_users)")
    parser.add_argument("--global_loc_map",  type=str,   default=None,
                        help="path to global_loc_map.pkl (default: ./data/global_loc_map.pkl)")
    parser.add_argument("--grid_map",        type=str,   default=None,
                        help="path to grid_map.pkl (default: ./data/grid_map.pkl)")
    parser.add_argument("--location_activity_map", type=str, default=None,
                        help="path to location_activity_map.pkl")
    parser.add_argument("--valid_frac",      type=float, default=0.2,
                        help="fraction of days held out for validation")
    parser.add_argument("--seed",            type=int,   default=42,
                        help="random seed for the train/valid day split")
    parser.add_argument("--model_dir",       type=str,   default=None,
                        help="directory to save checkpoints (default: ./predictors)")
    parser.add_argument("--save_path",       type=str,   default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args_train()
    train_and_validate(**vars(args))
