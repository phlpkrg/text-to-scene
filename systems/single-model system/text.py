import ast
import re
import yaml
import random
import os
import logging
import gdown
from collections import deque
from transformers import T5Tokenizer, T5ForConditionalGeneration


# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
MAX_PROMPT_TOKENS = 512
MAX_ASSETS_PER_ROOM = 10

AVAILABLE_ASSETS = [
    'Armchair', 'Bookshelf', 'Cabinet', 'CeilingLamp', 'Chair', 'ChaiseLongueSofa', 'ChineseChair',
    'ChildrenCabinet', 'CoffeTable', 'ConsoleTable', 'CornerSideTable', 'Desk', 'DiningChair',
    'DiningTable', 'DoubleBed', 'DressingChair', 'DressingTable', 'KidsBed', 'L-ShapedSofa',
    'LazySofa', 'LoungeChair', 'LoveseatSofa', 'Multi-SeatSofa', 'Nightstand', 'PendantLamp',
    'RoundEndTable', 'Shelf', 'SingleBed', 'Sofa', 'Stool', 'TVStand', 'Table', 'Wardrobe',
    'WineCabinet'
]

COMMON_ASSETS_BY_ROOM = {
    "LivingRoom": ["Sofa", "TVStand", "CoffeeTable", "Bookshelf", "Armchair"],
    "Bedroom": ["DoubleBed", "Wardrobe", "Nightstand", "DressingChair"],
    "Bathroom": ["Sink", "Toilet", "Shelf"],
    "Kitchen": ["DiningTable", "Chair", "Cabinet"],
    "DiningRoom": ["DiningChair", "DiningTable", "PendantLamp"],
    "Office": ["Desk", "Chair", "Shelf"],
    "Storage": ["Cabinet", "Shelf", "Wardrobe"]
}

MAX_ASSETS_BY_ROOM = {
    "LivingRoom": 8,
    "Bedroom": 6,
    "Bathroom": 4,
    "Kitchen": 6,
    "DiningRoom": 6,
    "Office": 5,
    "Storage": 4
}

# --- Single-model setup ---
LLM_CONFIG = {
    "LLM-Single": {"folder_id": "<paste gdrive folder id here>", "dir": "./LLM-Single"},
}

def download_folder_if_not_exists(folder_id, output_folder):
    logging.info(f"Downloading {output_folder} from Google Drive...")
    gdown.download_folder(
        f"https://drive.google.com/drive/folders/{folder_id}",
        output=output_folder,
        quiet=False,
        resume=True
    )

download_folder_if_not_exists(LLM_CONFIG["LLM-Single"]["folder_id"], LLM_CONFIG["LLM-Single"]["dir"])
llm_tokenizer = T5Tokenizer.from_pretrained(LLM_CONFIG["LLM-Single"]["dir"])
llm_model = T5ForConditionalGeneration.from_pretrained(LLM_CONFIG["LLM-Single"]["dir"])


# --- Input processing ---

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


# --- Generation and Parsing ---

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

def parse_llm_output(output: str):
    try:
        parsed = ast.literal_eval("{" + output + "}")
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"LLM output parsing failed: {e}")
    
    rooms = parsed.get("rooms", [])
    connections = parsed.get("connections", [])
    placements = {}

    for room, asset in parsed.get("assets", []):
        if room not in placements:
            placements[room] = []
        placements[room].append(case_shifter(asset))
    
    return rooms, connections, placements


# --- Validation ---

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
            queue.extend([n for n in graph[room] if n not in visited])
    return len(visited) == len(room_list)

def validate_output(rooms, connections, placements):
    if not is_graph_connected(rooms, connections):
        raise ValueError("Room connectivity invalid.")
    for room, assets in placements.items():
        if room not in rooms:
            raise ValueError(f"Unknown room in placements: {room}")
        for asset in assets:
            if asset not in AVAILABLE_ASSETS and asset != "Unknown":
                raise ValueError(f"Invalid asset: {asset}")


# --- Postprocessing ---

def normalize_room_type(room_name):
    return re.sub(r'\s+\d+$', '', room_name).replace(" ", "")

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


# --- Output formatter ---

def build_yaml(rooms, connections, placements):
    indexed_rooms = {i: placements.get(room, []) for i, room in enumerate(rooms)}
    return yaml.dump({
        "connections": connections,
        "room names": [room.rsplit(" ", 1)[0] for room in rooms],
        "rooms": indexed_rooms
    }, default_flow_style=False, sort_keys=False)


# --- Inference function ---

def infer(prompt):
    try:
        user_prompt, scaling = validate_and_preprocess_input(prompt, llm_tokenizer)
        logging.info(f"Validated prompt. Scaling: {scaling}")

        # Single query handles full scene
        llm_query = f"prompt='{user_prompt}', available={[a.lower() for a in AVAILABLE_ASSETS]}"
        llm_output = generate_text(llm_model, llm_tokenizer, llm_query)
        rooms, connections, placements = parse_llm_output(llm_output)

        validate_output(rooms, connections, placements)
        replace_unknowns(placements)
        scale_placements(scaling, placements)

        return build_yaml(rooms, connections, placements)

    except Exception as e:
        logging.error(f"Inference failed: {e}")
        return f"Error: {str(e)}"