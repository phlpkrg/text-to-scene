import ast
import re
import yaml
import random
import os
import logging
import gdown
from collections import deque
from transformers import T5Tokenizer, T5ForConditionalGeneration


# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# System-wide constants
MAX_PROMPT_TOKENS = 512
MAX_ASSETS_PER_ROOM = 10

# List of valid assets for placement
AVAILABLE_ASSETS = [
    'Armchair', 'Bookshelf', 'Cabinet', 'CeilingLamp', 'Chair', 'ChaiseLongueSofa', 'ChineseChair',
    'ChildrenCabinet', 'CoffeTable', 'ConsoleTable', 'CornerSideTable', 'Desk', 'DiningChair',
    'DiningTable', 'DoubleBed', 'DressingChair', 'DressingTable', 'KidsBed', 'L-ShapedSofa',
    'LazySofa', 'LoungeChair', 'LoveseatSofa', 'Multi-SeatSofa', 'Nightstand', 'PendantLamp',
    'RoundEndTable', 'Shelf', 'SingleBed', 'Sofa', 'Stool', 'TVStand', 'Table', 'Wardrobe',
    'WineCabinet'
]

# Common asset suggestions based on room type
COMMON_ASSETS_BY_ROOM = {
    "LivingRoom": ["Sofa", "TVStand", "CoffeeTable", "Bookshelf", "Armchair"],
    "Bedroom": ["DoubleBed", "Wardrobe", "Nightstand", "DressingChair"],
    "Bathroom": ["Sink", "Toilet", "Shelf"],
    "Kitchen": ["DiningTable", "Chair", "Cabinet"],
    "DiningRoom": ["DiningChair", "DiningTable", "PendantLamp"],
    "Office": ["Desk", "Chair", "Shelf"],
    "Storage": ["Cabinet", "Shelf", "Wardrobe"]
}

# Max asset limits per room type for scaling logic
MAX_ASSETS_BY_ROOM = {
    "LivingRoom": 8,
    "Bedroom": 6,
    "Bathroom": 4,
    "Kitchen": 6,
    "DiningRoom": 6,
    "Office": 5,
    "Storage": 4
}

# LLM download config
LLM_CONFIG = {
    "LM-Layouts": {"folder_id": "<paste gdrive folder id here>", "dir": "./LM-Layouts"},
    "LM-Placements": {"folder_id": "<paste gdrive folder id here>", "dir": "./LM-Placements"},
}

# Download models from Google Drive
def download_folder_if_not_exists(folder_id, output_folder):
    logging.info(f"Downloading {output_folder} from Google Drive...")
    gdown.download_folder(
        f"https://drive.google.com/drive/folders/{folder_id}",
        output=output_folder,
        quiet=False,
        resume=True
    )

for llm in LLM_CONFIG.values():
    download_folder_if_not_exists(llm["folder_id"], llm["dir"])

# Load tokenizer and model weights
llm1_tokenizer = T5Tokenizer.from_pretrained(LLM_CONFIG["LM-Layouts"]["dir"])
llm1_model = T5ForConditionalGeneration.from_pretrained(LLM_CONFIG["LM-Layouts"]["dir"])
llm2_tokenizer = T5Tokenizer.from_pretrained(LLM_CONFIG["LM-Placements"]["dir"])
llm2_model = T5ForConditionalGeneration.from_pretrained(LLM_CONFIG["LM-Placements"]["dir"])


# --- Input validation and preprocessing ---

def extract_scaling(prompt):
    match = re.search(r"(scaling)\s*=\s*([0-1](?:\.\d+)?)", prompt, re.IGNORECASE)
    if match:
        value = float(match.group(2))
        cleaned = re.sub(r"(scaling)\s*=\s*([0-1](?:\.\d+)?)(,)?", '', prompt, flags=re.IGNORECASE).strip().rstrip(",")
        return match.group(1), value, cleaned
    return None, 0.0, prompt

def validate_and_preprocess_input(prompt, tokenizer):
    if not prompt.strip():
        raise ValueError("Prompt is empty.")
    _, scaling, cleaned = extract_scaling(prompt)
    input_ids = tokenizer.encode(cleaned, return_tensors="pt", truncation=True)
    if input_ids.shape[1] > MAX_PROMPT_TOKENS:
        raise ValueError(f"Prompt too long. Max {MAX_PROMPT_TOKENS} tokens allowed.")
    return cleaned, scaling


# --- LLM generation wrapper ---

def generate_text(model, tokenizer, query):
    input_ids = tokenizer.encode(query, return_tensors="pt", max_length=512, truncation=True)
    output_ids = model.generate(
        input_ids,
        max_length=512,
        num_beams=5,
        temperature=0.5,
        top_k=50,
        top_p=0.9,
        do_sample=True,
        early_stopping=True
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# --- LLM output parsers ---

def parse_llm1_output(data_str):
    try:
        data_dict = ast.literal_eval("{" + data_str + "}")
        return data_dict.get('rooms', []), data_dict.get('connections', [])
    except Exception as e:
        logging.error(f"LLM1 parsing failed: {e}")
        return [], []

def parse_llm2_output(data_str):
    data_dict = {}
    room_entries = re.split(r',\s*(?![^\[]*\])', data_str)
    for entry in room_entries:
        key, value = entry.split("=", 1)
        key, value = key.strip(), eval(value.strip())
        data_dict[key] = [case_shifter(asset) for asset in value]
    return data_dict


# --- Output validation (graph & assets) ---

def is_graph_connected(room_list, connection_list):
    if not room_list:
        return False
    graph = {room: [] for room in room_list}
    for a, b in connection_list:
        graph[a].append(b)
        graph[b].append(a)

    visited = set()
    queue = deque([room_list[0]])
    while queue:
        room = queue.popleft()
        if room not in visited:
            visited.add(room)
            queue.extend([neighbor for neighbor in graph[room] if neighbor not in visited])
    return len(visited) == len(room_list)

def validate_output(rooms, connections, placements):
    if not is_graph_connected(rooms, connections):
        raise ValueError("Room connectivity invalid — graph is not fully connected.")

    for room, assets in placements.items():
        if room not in rooms:
            raise ValueError(f"Invalid room in placements: {room}")
        for asset in assets:
            if asset not in AVAILABLE_ASSETS and asset != "Unknown":
                raise ValueError(f"Invalid asset: '{asset}' in room '{room}'.")


# --- Postprocessing (fallbacks + scaling) ---

def case_shifter(asset_name):
    formatted = asset_name.replace(" ", "").lower()
    for asset in AVAILABLE_ASSETS:
        if formatted == asset.replace(" ", "").lower():
            return asset
    return "Unknown"

def replace_unknowns(placements):
    for room_name, assets in placements.items():
        base_type = normalize_room_type(room_name)
        replacement_pool = COMMON_ASSETS_BY_ROOM.get(base_type, AVAILABLE_ASSETS)

        placements[room_name] = [
            random.choice(replacement_pool) if asset == "Unknown" else asset
            for asset in assets
        ]

# Normalize room name to detect its type (e.g., "Bedroom 2" → "Bedroom")
def normalize_room_type(room_name):
    return re.sub(r'\s+\d+$', '', room_name).replace(" ", "")

# Apply scaling using predefined asset templates per room type
def scale_placements(scaling, placements):
    for room_name in placements:
        base_type = normalize_room_type(room_name)
        common_assets = COMMON_ASSETS_BY_ROOM.get(base_type, AVAILABLE_ASSETS)
        max_assets = MAX_ASSETS_BY_ROOM.get(base_type, MAX_ASSETS_PER_ROOM)

        current_assets = placements[room_name]
        num_to_add = int(max_assets * scaling)
        num_to_fill = max(0, num_to_add - len(current_assets))

        extra_assets = random.sample(common_assets, min(num_to_fill, len(common_assets)))
        placements[room_name].extend(extra_assets)
    return placements


# --- YAML scene graph formatter ---

def build_yaml(rooms, connections, placements):
    indexed_rooms = {i: placements.get(room, []) for i, room in enumerate(rooms)}
    yaml_data = {
        "connections": connections,
        "room names": [room.rsplit(" ", 1)[0] for room in rooms],
        "rooms": indexed_rooms
    }
    return yaml.dump(yaml_data, default_flow_style=False, sort_keys=False)


# --- Main pipeline ---

def infer(prompt):
    try:
        user_prompt, scaling = validate_and_preprocess_input(prompt, llm1_tokenizer)
        logging.info(f"Validated prompt. Scaling: {scaling}")

        llm1_output = generate_text(llm1_model, llm1_tokenizer, user_prompt)
        rooms, connections = parse_llm1_output(llm1_output)

        llm2_query = f"prompt='{user_prompt}', rooms={str(rooms)}, available={[a.lower() for a in AVAILABLE_ASSETS]}"
        llm2_output = generate_text(llm2_model, llm2_tokenizer, llm2_query)
        placements = parse_llm2_output(llm2_output)

        validate_output(rooms, connections, placements)
        replace_unknowns(placements)
        scale_placements(scaling, placements)

        return build_yaml(rooms, connections, placements)

    except Exception as e:
        logging.error(f"Inference failed: {e}")
        return f"Error: {str(e)}"