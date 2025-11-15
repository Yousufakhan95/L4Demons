# ============================================================
#  CNN-BASED CHESSHACKS BOT + TRAINING ENGINE (DYNAMIC DEPTH)
#  - CNN value net over 8x8 planes
#  - Negamax + alpha-beta + quiescence
#  - Dynamic search depth for actual games
#  - Supervised training from PGN/CSV with Elo weighting
#  - Self-play + games vs Stockfish
#  - Explicit backprop_step() for training (Ctrl-C to stop)
# ============================================================

from .utils import chess_manager, GameContext  # adapt if your path is different
from chess import Board, Move
import chess
import chess.engine
import chess.pgn
import random
import math
import os
import csv
import numpy as np
from typing import Optional, List, Tuple
import concurrent.futures

# ============================================================
#  TORCH SETUP
# ============================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception as e:
    print("[WARN] Torch not available:", e)
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

# ============================================================
#  CONSTANTS + PATHS
# ============================================================

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # project root

BOARD_SIZE = 8
NUM_PIECE_PLANES = 12   # 6 piece types per color
NUM_EXTRA_PLANES = 1    # side-to-move plane
NUM_PLANES = NUM_PIECE_PLANES + NUM_EXTRA_PLANES

# We keep 4 extra scalar features (castling rights) at the end of the vector
FEATURE_DIM = NUM_PLANES * BOARD_SIZE * BOARD_SIZE + 4

MODEL = None
DEVICE = "cuda"
MODEL_PATH = os.path.join(ROOT_DIR,"weights", "model.pt")
BEST_MODEL_PATH = os.path.join(ROOT_DIR, "weights", "model_best.pt")

# ---------------------------
# Stockfish config
# ---------------------------
STOCKFISH_PATH = os.path.join(ROOT_DIR, "engines", "stockfish.exe")
STOCKFISH_TIME_LIMIT = 0.03  # seconds per move for training/eval
STOCKFISH_MAX_MOVES = 200

# Elo ladder config
EVAL_ELOS = [1000]
EVAL_GAMES_PER_LEVEL = 4
EVAL_LADDER_INTERVAL_GAMES = 20  # run ladder every N self-play games

# ---------------------------
# Search config
# ---------------------------
# Base depth for online play (you can tweak this):
BASE_SEARCH_DEPTH = 2          # starting point
MIN_SEARCH_DEPTH = 1          # clamp
MAX_SEARCH_DEPTH = 4           # clamp

SELFPLAY_DEPTH = 2             # depth in self-play for speed
TRAIN_SEARCH_DEPTH = 3         # depth vs Stockfish in training

# ---------------------------
# Data config
# ---------------------------
MIN_FULLMOVES_FOR_PGN_LABEL = 10  # start labeling after move 10

# ---------------------------
# Elo weighting config
# ---------------------------
# Average Elo between players is mapped to a [0.1, 1.0] weight.
# Below MIN_AVG_ELO_FOR_WEIGHT → use 0.1; above MAX → use 1.0.
MIN_AVG_ELO_FOR_WEIGHT = 1400
MAX_AVG_ELO_FOR_WEIGHT = 2400


# ============================================================
#  FEATURE ENCODING
# ============================================================

def piece_plane_index(piece: chess.Piece) -> int:
    # 0–5 white pawn..king, 6–11 black pawn..king
    return (0 if piece.color else 6) + (piece.piece_type - 1)


def board_to_tensor(board: chess.Board):
    """
    Encode board into a 1D tensor of size FEATURE_DIM.
    Layout:
      - First NUM_PLANES * 64 entries: 8x8 planes (piece planes + side-to-move)
      - Last 4 entries: castling rights flags.
    """
    if not TORCH_AVAILABLE or torch is None:
        return None

    planes = torch.zeros(NUM_PLANES, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)

    for square, piece in board.piece_map().items():
        rank = square // 8
        file = square % 8
        planes[piece_plane_index(piece), rank, file] = 1.0

    # side-to-move plane (all ones if white to move, zeros if black)
    planes[NUM_PIECE_PLANES, :, :] = 1.0 if board.turn else 0.0

    flat = planes.reshape(-1)

    castle_flags = torch.tensor([
        float(board.has_kingside_castling_rights(True)),
        float(board.has_queenside_castling_rights(True)),
        float(board.has_kingside_castling_rights(False)),
        float(board.has_queenside_castling_rights(False)),
    ], dtype=torch.float32)

    return torch.cat([flat, castle_flags], dim=0)


# ============================================================
#  CNN VALUE NETWORK
# ============================================================

if TORCH_AVAILABLE and nn is not None:

    class CnnValueNet(nn.Module):
        """
        CNN over board planes + small dense head with extra meta features.
        Input: flat vector length FEATURE_DIM
        Output: scalar value in [-1, 1] from White's POV for given board.
        """
        def __init__(self, input_dim: int, num_planes: int = NUM_PLANES):
            super().__init__()
            self.num_planes = num_planes
            board_vec_dim = num_planes * BOARD_SIZE * BOARD_SIZE
            self.board_vec_dim = board_vec_dim
            self.meta_dim = input_dim - board_vec_dim  # should be 4

            # Convolutional trunk over 8x8 planes
            self.conv1 = nn.Conv2d(num_planes, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

            conv_output_dim = 64 * BOARD_SIZE * BOARD_SIZE  # 64 channels * 8 * 8

            self.fc1 = nn.Linear(conv_output_dim + self.meta_dim, 512)
            self.fc2 = nn.Linear(512, 128)
            self.fc_out = nn.Linear(128, 1)
            self.dropout = nn.Dropout(0.2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [batch, FEATURE_DIM]
            board_flat = x[:, :self.board_vec_dim]
            meta = x[:, self.board_vec_dim:]  # [batch, meta_dim]

            board_planes = board_flat.view(-1, self.num_planes, BOARD_SIZE, BOARD_SIZE)

            h = F.relu(self.conv1(board_planes))
            h = F.relu(self.conv2(h))
            h = F.relu(self.conv3(h))

            h = h.view(h.size(0), -1)
            h = torch.cat([h, meta], dim=1)

            h = F.relu(self.fc1(h))
            h = self.dropout(h)
            h = F.relu(self.fc2(h))
            out = torch.tanh(self.fc_out(h)).squeeze(-1)
            return out

else:
    CnnValueNet = None


def init_model():
    """
    Init global MODEL (CNN) and load weights if model.pt exists.
    """
    global MODEL, DEVICE

    if not TORCH_AVAILABLE or CnnValueNet is None:
        MODEL = None
        print("[ML] Torch or CNN class not available, running without ML.")
        return

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ML] Using device: {DEVICE}")

    MODEL = CnnValueNet(FEATURE_DIM).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        try:
            state = torch.load(MODEL_PATH, map_location=DEVICE)
            # Allow older formats that might wrap state dict
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            MODEL.load_state_dict(state)
            print(f"[ML] Loaded model from {MODEL_PATH}")
        except Exception as e:
            print(f"[ML] Failed to load model.pt, using fresh weights: {e}")
    else:
        print("[ML] No model.pt found — starting from scratch.")

    MODEL.eval()


# Initialize immediately so the dashboard can use the bot
init_model()


# ============================================================
#  SIMPLE EVAL + NN-BASED EVAL
# ============================================================

# Basic material values in centipawns
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   0,  # king safety handled separately if needed
}


def material_eval_cp(board: chess.Board) -> int:
    score = 0
    for square, piece in board.piece_map().items():
        val = PIECE_VALUES.get(piece.piece_type, 0)
        if piece.color == chess.WHITE:
            score += val
        else:
            score -= val
    return score


def simple_classical_eval(board: chess.Board) -> float:
    """
    Very simple classical eval in [-1, 1], from side-to-move POV.
    We mostly rely on CNN; this is just a stabilizer.
    """
    cp = material_eval_cp(board)

    # Normalize to [-1, 1] assuming ±2000cp extremes
    MAX_CP = 2000.0
    cp = max(-MAX_CP, min(MAX_CP, float(cp)))
    val = cp / MAX_CP

    # side-to-move POV
    if board.turn == chess.WHITE:
        return val
    else:
        return -val


def evaluate_position(board: chess.Board) -> float:
    """
    Final evaluation from POV of side to move, in [-1, 1].

    Combination:
      - light classical eval
      - CNN value net (stronger pattern recognition)
    """
    classical = simple_classical_eval(board)

    if not TORCH_AVAILABLE or MODEL is None:
        return classical

    x = board_to_tensor(board)
    if x is None:
        return classical

    with torch.no_grad():
        x = x.to(DEVICE).unsqueeze(0)
        v = MODEL(x)[0].item()  # NN output in [-1, 1]

    # Blend weights. You can tune these.
    ALPHA = 0.2  # classical
    BETA = 0.8   # CNN

    return ALPHA * classical + BETA * v


# ============================================================
#  DYNAMIC DEPTH FOR ACTUAL GAMES
# ============================================================

def estimate_material_phase(board: chess.Board) -> float:
    """
    Rough material count in "pawns" for both sides combined.
    Used to guess opening / middlegame / endgame.
    """
    total_cp = 0
    for _, piece in board.piece_map().items():
        total_cp += PIECE_VALUES.get(piece.piece_type, 0)
    # pawns-equivalent
    return total_cp / 100.0


def get_dynamic_depth(board: chess.Board) -> int:
    """
    Dynamic depth for actual dashboard games.

    Heuristics:
      - Start from BASE_SEARCH_DEPTH
      - Increase depth in:
          * Endgames (low material)
          * Later in the game (fullmove_number big)
          * When side to move is in check (tactical)
      - Clamp between MIN_SEARCH_DEPTH and MAX_SEARCH_DEPTH
    """
    depth = BASE_SEARCH_DEPTH

    phase_pawns = estimate_material_phase(board)  # ~32 at start
    moves = board.fullmove_number
    in_check = board.is_check()

    # Later game or low material → think deeper
    if moves >= 25 or phase_pawns <= 18:
        depth += 1
    if moves >= 40 or phase_pawns <= 12:
        depth += 1

    # If the side to move is in check, push a little deeper
    if in_check:
        depth += 1

    # Clamp
    depth = max(MIN_SEARCH_DEPTH, min(MAX_SEARCH_DEPTH, depth))
    return depth


# ============================================================
#  NEGAMAX + ALPHA-BETA + QUIESCENCE
# ============================================================

def softmax(scores: List[float]) -> List[float]:
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(scores)] * len(scores)
    return [e / total for e in exps]


def quiescence(board: chess.Board, alpha: float, beta: float, depth_cap: int = 4) -> float:
    """
    Very small quiescence search:
      - Only explore capture moves (and promotions)
      - Limit depth to avoid explosion.
    """
    stand_pat = evaluate_position(board)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    if depth_cap <= 0:
        return alpha

    for move in board.generate_legal_moves():
        if not board.is_capture(move) and not move.promotion:
            continue
        board.push(move)
        score = -quiescence(board, -beta, -alpha, depth_cap - 1)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def negamax(board: chess.Board, depth: int, alpha: float, beta: float) -> float:
    """
    Negamax with alpha-beta pruning and quiescence at leaf nodes.
    """
    if depth == 0 or board.is_game_over():
        return quiescence(board, alpha, beta)

    max_eval = -float("inf")
    legal_moves = list(board.generate_legal_moves())
    # Optional: simple move ordering (captures first)
    legal_moves.sort(key=lambda m: board.is_capture(m), reverse=True)

    for move in legal_moves:
        board.push(move)
        eval_score = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()

        if eval_score > max_eval:
            max_eval = eval_score

        if eval_score > alpha:
            alpha = eval_score
        if alpha >= beta:
            break  # cutoff

    return max_eval


def choose_move_with_search(
    board: chess.Board,
    legal_moves,
    depth: Optional[int] = None,
):
    """
    Use negamax + alpha-beta to pick the best move.

    If depth is None:
      - For *actual games* (dashboard), we call this with depth=None and
        dynamically compute depth with get_dynamic_depth(board).
    """
    if not legal_moves:
        return None, {}

    # If no NN available, fall back to random
    if not TORCH_AVAILABLE or MODEL is None:
        n = len(legal_moves)
        return random.choice(legal_moves), {m: 1.0 / n for m in legal_moves}

    if depth is None:
        depth = get_dynamic_depth(board)

    best_score = -float("inf")
    best_move = None
    scores = []

    for move in legal_moves:
        board.push(move)
        score = -negamax(board, depth - 1, -float("inf"), float("inf"))
        board.pop()

        scores.append(score)
        if best_move is None or score > best_score:
            best_score = score
            best_move = move

    probs = softmax(scores)
    prob_map = {m: p for m, p in zip(legal_moves, probs)}
    return best_move, prob_map


# ============================================================
#  CHESSHACKS ENTRYPOINTS (ACTUAL GAMES)
# ============================================================

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    """
    Called for every move request by ChessHacks.

    - Uses dynamic depth for search (get_dynamic_depth).
    - Depth scales with phase of game + checks.
    """
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available.")

    # depth=None → use dynamic depth
    move, prob_map = choose_move_with_search(ctx.board, legal_moves, depth=None)
    ctx.logProbabilities(prob_map)
    return move


@chess_manager.reset
def reset_func(ctx: GameContext):
    """
    Called when a new game starts.
    """
    pass


# ============================================================
#  ELO WEIGHT HELPER
# ============================================================

def elo_str_to_int(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def compute_elo_weight(avg_elo: Optional[float]) -> float:
    """
    Map average Elo to a weight in [0.1, 1.0].
    - If Elo is unknown → 1.0
    - Below MIN_AVG_ELO_FOR_WEIGHT → 0.1
    - Above MAX_AVG_ELO_FOR_WEIGHT → 1.0
    - Linear in between.
    """
    if avg_elo is None:
        return 1.0

    lo = MIN_AVG_ELO_FOR_WEIGHT
    hi = MAX_AVG_ELO_FOR_WEIGHT

    if avg_elo <= lo:
        return 0.1
    if avg_elo >= hi:
        return 1.0

    t = (avg_elo - lo) / (hi - lo)  # 0..1
    return 0.1 + 0.9 * t


# ============================================================
#  PGN / CSV DATA LOADERS (SUPERVISED TRAINING, ELO-WEIGHTED)
# ============================================================

def game_result_to_value(result: str) -> float:
    if result == "1-0":
        return 1.0
    elif result == "0-1":
        return -1.0
    else:
        return 0.0


def load_pgn_dataset(path: str, max_positions: Optional[int] = None):
    """
    Load PGN file and generate (x, y, w) triples:
      - x = feature tensor
      - y = value from POV of side-to-move in that position
      - w = sample weight based on average Elo of players
    """
    if not TORCH_AVAILABLE or torch is None:
        return []

    basename = os.path.basename(path)
    print(f"[DATA] Loading PGN: {basename}")
    positions = []
    game_count = 0

    with open(path, "r", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            game_count += 1

            headers = game.headers

            # Try to fetch Elo from typical PGN tags
            w_elo = (
                elo_str_to_int(headers.get("WhiteElo"))
                or elo_str_to_int(headers.get("WhiteRating"))
            )
            b_elo = (
                elo_str_to_int(headers.get("BlackElo"))
                or elo_str_to_int(headers.get("BlackRating"))
            )

            avg_elo = None
            if w_elo is not None and b_elo is not None:
                avg_elo = (w_elo + b_elo) / 2.0
            elif w_elo is not None:
                avg_elo = w_elo
            elif b_elo is not None:
                avg_elo = b_elo

            sample_weight = compute_elo_weight(avg_elo)

            result = headers.get("Result", "*")
            if result == "1-0":
                winner_color = chess.WHITE
            elif result == "0-1":
                winner_color = chess.BLACK
            else:
                winner_color = None  # draw or unknown

            board = game.board()
            for mv in game.mainline_moves():
                board.push(mv)

                if board.fullmove_number < MIN_FULLMOVES_FOR_PGN_LABEL:
                    continue

                x = board_to_tensor(board)
                if x is None:
                    continue

                if winner_color is None:
                    val = 0.0
                else:
                    val = 1.0 if board.turn == winner_color else -1.0

                positions.append((x, val, sample_weight))

                if max_positions is not None and len(positions) >= max_positions:
                    print(f"[DATA] PGN {basename}: {len(positions)} positions (truncated).")
                    return positions

    print(f"[DATA] PGN {basename}: {game_count} games, {len(positions)} positions.")
    return positions


def load_lichess_games_csv(path: str, max_positions: Optional[int] = None):
    """
    Load a Lichess games CSV with columns including:
      id, winner, moves, white_elo / black_elo or similar.

    Outputs (x, y, w) triples with Elo-based weights.
    """
    if not TORCH_AVAILABLE or torch is None:
        print("[DATA] Torch not available; cannot load Lichess CSV.")
        return []

    basename = os.path.basename(path)
    print(f"[DATA] Loading Lichess CSV: {basename}")
    positions = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # winner
            winner = row.get("winner", "").strip().lower()
            if winner == "white":
                winner_color = chess.WHITE
            elif winner == "black":
                winner_color = chess.BLACK
            else:
                winner_color = None

            # Try a few common rating column names
            w_elo = (
                elo_str_to_int(row.get("white_elo"))
                or elo_str_to_int(row.get("white_rating"))
                or elo_str_to_int(row.get("whiteelo"))
            )
            b_elo = (
                elo_str_to_int(row.get("black_elo"))
                or elo_str_to_int(row.get("black_rating"))
                or elo_str_to_int(row.get("blackelo"))
            )

            avg_elo = None
            if w_elo is not None and b_elo is not None:
                avg_elo = (w_elo + b_elo) / 2.0
            elif w_elo is not None:
                avg_elo = w_elo
            elif b_elo is not None:
                avg_elo = b_elo

            sample_weight = compute_elo_weight(avg_elo)

            moves_str = row.get("moves", "")
            if not moves_str:
                continue

            board = chess.Board()
            moves = moves_str.split()
            for mv in moves:
                try:
                    board.push_san(mv)
                except ValueError:
                    break  # invalid SAN -> skip game

                if board.fullmove_number < MIN_FULLMOVES_FOR_PGN_LABEL:
                    continue

                x = board_to_tensor(board)
                if x is None:
                    continue

                if winner_color is None:
                    val = 0.0
                else:
                    val = 1.0 if board.turn == winner_color else -1.0

                positions.append((x, val, sample_weight))

                if max_positions is not None and len(positions) >= max_positions:
                    print(f"[DATA] CSV {basename}: {len(positions)} positions (truncated).")
                    return positions

    print(f"[DATA] CSV {basename}: {len(positions)} positions.")
    return positions


def load_all_datasets(folder: str = "datasets", max_positions: Optional[int] = None):
    """
    Load everything in datasets folder:
      - *.pgn -> PGN games
      - lichess-style *.csv -> moves/winner
    Returns list of (x, y, weight).
    """
    if not TORCH_AVAILABLE:
        print("[DATA] Torch not available; skipping dataset loading.")
        return []

    if not os.path.isabs(folder):
        folder = os.path.join(ROOT_DIR, folder)

    if not os.path.exists(folder):
        print(f"[DATA] No dataset folder: {folder}")
        return []

    files = os.listdir(folder)
    pgn_files = [f for f in files if f.lower().endswith(".pgn")]
    csv_files = [f for f in files if f.lower().endswith(".csv")]

    all_positions: List[Tuple[torch.Tensor, float, float]] = []

    # Parallel PGN loading
    if pgn_files:
        print(f"[DATA] Parallel loading {len(pgn_files)} PGN files...")
        total = len(pgn_files)
        completed = 0

        if max_positions is not None:
            per_file_limit = max(1, max_positions // max(total, 1))
        else:
            per_file_limit = None

        with concurrent.futures.ThreadPoolExecutor() as ex:
            futures = {}
            for f_name in pgn_files:
                path = os.path.join(folder, f_name)
                futures[ex.submit(load_pgn_dataset, path, per_file_limit)] = f_name

            for fut in concurrent.futures.as_completed(futures):
                fname = futures[fut]
                try:
                    pos = fut.result()
                    all_positions.extend(pos)
                    print(f"[DATA] PGN {fname}: {len(pos)} positions")
                except Exception as e:
                    print(f"[DATA] Failed to load PGN {fname}: {e}")

                completed += 1
                print(f"[DATA] PGN progress: {completed}/{total}")

                if max_positions is not None and len(all_positions) >= max_positions:
                    all_positions = all_positions[:max_positions]
                    print(f"[DATA] Reached max_positions={max_positions} from PGNs.")
                    csv_files = []
                    break

    # CSV loading
    for f_name in csv_files:
        path = os.path.join(folder, f_name)
        pos = load_lichess_games_csv(path, max_positions)
        all_positions.extend(pos)
        if max_positions is not None and len(all_positions) >= max_positions:
            all_positions = all_positions[:max_positions]
            break

    print(f"[DATA] Loaded {len(all_positions)} total positions from {folder}.")
    return all_positions


# ============================================================
#  REPLAY BUFFER + SELF-PLAY + STOCKFISH GAMES
# ============================================================

if TORCH_AVAILABLE and CnnValueNet is not None:

    class ReplayBuffer:
        def __init__(self, cap: int = 500_000):
            self.cap = cap
            # store (x, y, weight)
            self.data: List[Tuple[torch.Tensor, float, float]] = []

        def add(self, x, y, weight: float = 1.0):
            if len(self.data) >= self.cap:
                self.data.pop(0)
            self.data.append((x, float(y), float(weight)))

        def extend(self, items):
            for item in items:
                if len(item) == 3:
                    x, y, w = item
                else:
                    # backward compatibility if something passes (x, y)
                    x, y = item
                    w = 1.0
                self.add(x, y, w)

        def sample(self, n: int):
            batch = random.sample(self.data, min(n, len(self.data)))
            xs, ys, ws = zip(*batch)
            xs = torch.stack(xs).to(DEVICE)
            ys = torch.tensor(ys, dtype=torch.float32).to(DEVICE)
            ws = torch.tensor(ws, dtype=torch.float32).to(DEVICE)
            return xs, ys, ws

        def __len__(self):
            return len(self.data)

    REPLAY = ReplayBuffer()
else:
    REPLAY = None


def choose_move_for_selfplay(board: Board, depth: int) -> Optional[Move]:
    legal = list(board.generate_legal_moves())
    if not legal:
        return None
    move, _ = choose_move_with_search(board, legal_moves=legal, depth=depth)
    return move


def self_play_game(max_moves: int = 300, depth: int = SELFPLAY_DEPTH):
    """
    Self-play game with the current MODEL.

    We "backpropagate" the game result to all positions in the trajectory:
      - For each position we store (x, y, 1.0) where y is +1/-1/0 from POV of
        side to move in that position.
    """
    if MODEL is None or not TORCH_AVAILABLE:
        return []

    board = Board()
    traj = []

    for _ in range(max_moves):
        if board.is_game_over():
            break
        x = board_to_tensor(board)
        if x is not None:
            traj.append((board.turn, x))

        mv = choose_move_for_selfplay(board, depth=depth)
        if mv is None:
            break
        board.push(mv)

    res = board.result(claim_draw=True)
    if res == "1-0":
        w, b = 1.0, -1.0
    elif res == "0-1":
        w, b = -1.0, 1.0
    else:
        w = b = 0.0

    labeled = [(x, w if turn else b, 1.0) for (turn, x) in traj]
    return labeled


# --------------------------- Stockfish wrapper ---------------------------

def create_stockfish_engine(sf_elo: Optional[int] = 1000, sf_skill: Optional[int] = None):
    """
    Launch Stockfish with optional strength limiting via:
      - UCI_LimitStrength + UCI_Elo (preferred)
      - Skill Level (0–20)
    Returns (engine, ok_flag).
    """
    if not os.path.exists(STOCKFISH_PATH):
        print(f"[SF] Stockfish not found at: {STOCKFISH_PATH}")
        return None, False

    try:
        engine = chess.engine.SimpleEngine.popen_uci([STOCKFISH_PATH])
        print("[SF] Stockfish engine started.")
    except Exception as e:
        print(f"[SF] Error launching Stockfish: {e}")
        return None, False

    try:
        opts = engine.options
    except Exception as e:
        print(f"[SF] Could not read engine options: {e}")
        return engine, True

    # Elo limiting
    if sf_elo is not None and "UCI_LimitStrength" in opts and "UCI_Elo" in opts:
        elo_opt = opts["UCI_Elo"]
        min_elo = getattr(elo_opt, "min", 1320)
        max_elo = getattr(elo_opt, "max", 3000)
        requested = int(sf_elo)
        clamped = max(min_elo, min(requested, max_elo))
        cfg = {"UCI_LimitStrength": True, "UCI_Elo": clamped}
        try:
            engine.configure(cfg)
            if clamped != requested:
                print(f"[SF] Requested Elo {requested}, clamped to [{min_elo},{max_elo}] → {clamped}.")
            else:
                print(f"[SF] Configured Stockfish Elo {clamped}.")
            return engine, True
        except Exception as e:
            print(f"[SF] Failed Elo config {cfg}: {e}")

    # Skill Level fallback
    if "Skill Level" in opts:
        if sf_skill is None:
            sf_skill = 0  # easiest
        skill = max(0, min(20, int(sf_skill)))
        cfg = {"Skill Level": skill}
        try:
            engine.configure(cfg)
            print(f"[SF] Configured Stockfish Skill Level = {skill}.")
            return engine, True
        except Exception as e:
            print(f"[SF] Failed Skill Level config {cfg}: {e}")

    print("[SF] Running Stockfish at full strength (no limiting).")
    return engine, True


def game_vs_stockfish(
    engine,
    our_color=chess.WHITE,
    max_moves: int = STOCKFISH_MAX_MOVES,
    our_depth: int = TRAIN_SEARCH_DEPTH,
):
    """
    Play one game vs Stockfish. Returns (labeled_positions, result_from_our_pov).
    Labeled positions include weight=1.0 (no Elo here).
    """
    if MODEL is None or not TORCH_AVAILABLE:
        return [], 0.0

    board = Board()
    traj = []

    for _ in range(max_moves):
        if board.is_game_over():
            break

        x = board_to_tensor(board)
        if x is not None:
            traj.append((board.turn, x))

        if board.turn == our_color:
            mv = choose_move_for_selfplay(board, depth=our_depth)
        else:
            try:
                result = engine.play(board, chess.engine.Limit(time=STOCKFISH_TIME_LIMIT))
                mv = result.move
            except Exception as e:
                print("[SF] Engine error:", e)
                break

        if mv is None:
            break
        board.push(mv)

    res = board.result(claim_draw=True)
    if res == "1-0":
        w, b = 1.0, -1.0
    elif res == "0-1":
        w, b = -1.0, 1.0
    else:
        w = b = 0.0

    labeled = [(x, w if turn else b, 1.0) for (turn, x) in traj]

    if res == "1-0":
        result_value = 1.0 if our_color == chess.WHITE else -1.0
    elif res == "0-1":
        result_value = 1.0 if our_color == chess.BLACK else -1.0
    else:
        result_value = 0.0

    return labeled, result_value


# ============================================================
#  BACKPROP STEP + TRAINING LOOP
# ============================================================

def backprop_step(optimizer, batch_size: int = 512):
    """
    One explicit backpropagation step:
      - Sample a mini-batch from the replay buffer
      - Compute weighted MSE loss between NN predictions and target values
        using Elo-based sample weights:
            loss = mean( weight_i * (pred_i - y_i)^2 )
      - Backpropagate gradients and update weights
    """
    if MODEL is None or REPLAY is None or len(REPLAY) == 0:
        return None

    MODEL.train()
    xs, ys, ws = REPLAY.sample(batch_size)
    preds = MODEL(xs)
    loss_vec = (preds - ys) ** 2

    # Normalize weights so their mean is ~1, so LR stays sane
    ws = ws / (ws.mean() + 1e-8)
    loss = (ws * loss_vec).mean()

    optimizer.zero_grad()
    loss.backward()   # <-- backpropagation happens here
    optimizer.step()
    MODEL.eval()
    return float(loss.item())


def run_stockfish_elo_ladder(eval_elos=None, games_per_level: int = EVAL_GAMES_PER_LEVEL):
    if MODEL is None or not TORCH_AVAILABLE:
        print("[EVAL] No model / torch for ladder.")
        return

    if eval_elos is None:
        eval_elos = EVAL_ELOS

    print("\n[EVAL] ===== Stockfish Elo ladder evaluation =====")
    for elo in eval_elos:
        engine, ok = create_stockfish_engine(sf_elo=elo)
        if not ok or engine is None:
            print(f"[EVAL] Could not start Stockfish {elo}. Skipping.")
            continue

        wins = draws = losses = 0
        for i in range(games_per_level):
            color = chess.WHITE if i % 2 == 0 else chess.BLACK
            _, res_val = game_vs_stockfish(engine, our_color=color)
            if res_val > 0:
                wins += 1
            elif res_val < 0:
                losses += 1
            else:
                draws += 1
            print(f"[EVAL] Elo {elo} — game {i+1}/{games_per_level}: result={res_val}")

        try:
            engine.quit()
        except Exception:
            pass

        played = wins + draws + losses
        if played == 0:
            print(f"[EVAL] Elo {elo}: no completed games.")
            continue

        score = (wins + 0.5 * draws) / played
        if score <= 0.0 or score >= 1.0:
            perf_str = "undefined (score too extreme)"
        else:
            perf = elo + 400 * math.log10(score / (1 - score))
            perf_str = f"≈ {perf:.0f}"

        print(f"[EVAL] vs SF {elo}: {wins}-{losses}-{draws} "
              f"(score={score:.3f}) → performance {perf_str}")
    print("[EVAL] ==========================================\n")


def train_forever(
    dataset_folder: str = "datasets",
    max_dataset_positions: Optional[int] = None,
    batch_size: int = 512,
    base_lr: float = 1e-3,
    selfplay_games_per_cycle: int = 10,
):
    """
    Main training driver.
    - Load supervised data once into replay (Elo-weighted).
    - Infinite cycles:
        - many self-play games
        - a few games vs Stockfish
        - multiple backprop_step() calls per cycle
    - Periodically runs an Elo ladder.
    - Ctrl-C to stop; final/best model saved.
    """
    if not TORCH_AVAILABLE or CnnValueNet is None:
        print("[TRAIN] Torch or CNN not available, cannot train.")
        return

    global MODEL
    if MODEL is None:
        init_model()
        if MODEL is None:
            print("[TRAIN] No model instance; aborting.")
            return

    optimizer = torch.optim.Adam(MODEL.parameters(), lr=base_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Supervised data
    data = load_all_datasets(dataset_folder, max_dataset_positions)
    if data:
        random.shuffle(data)
        if REPLAY is not None:
            REPLAY.extend(data)
            print(f"[TRAIN] Replay size after supervised load: {len(REPLAY)}")
    else:
        print("[TRAIN] No supervised dataset loaded; relying on self-play + Stockfish only.")

    best_val_like_loss = None
    best_state = None
    total_selfplay_games = 0

    def estimate_val_like_loss(num_samples: int = 4096):
        if REPLAY is None or len(REPLAY) == 0:
            return None
        MODEL.eval()
        with torch.no_grad():
            xs, ys, ws = REPLAY.sample(min(num_samples, len(REPLAY)))
            preds = MODEL(xs)
            # approximate validation: ignore weights here, it's just a metric
            loss = F.mse_loss(preds, ys)
        return float(loss.item())

    # Stockfish engine for training games
    sf_engine, sf_ok = create_stockfish_engine(sf_elo=1000)

    try:
        cycle = 0
        while True:
            cycle += 1
            print(f"\n[TRAIN] ===== Cycle {cycle} =====")

            # --- Self-play ---
            for g in range(selfplay_games_per_cycle):
                positions = self_play_game()
                if REPLAY is not None:
                    REPLAY.extend(positions)
                total_selfplay_games += 1
                print(f"[TRAIN] Self-play game {total_selfplay_games}: {len(positions)} positions, "
                      f"replay size={len(REPLAY)}")

            # --- Games vs Stockfish ---
            if sf_ok and sf_engine is not None:
                for i in range(2):  # two games per cycle (white & black)
                    color = chess.WHITE if i % 2 == 0 else chess.BLACK
                    pos_sf, res = game_vs_stockfish(sf_engine, our_color=color)
                    if REPLAY is not None:
                        REPLAY.extend(pos_sf)
                    print(f"[TRAIN] vs Stockfish ({'White' if color == chess.WHITE else 'Black'}): "
                          f"{len(pos_sf)} positions, result={res}, replay={len(REPLAY)}")

            # --- Backprop steps ---
            steps = max(10, selfplay_games_per_cycle * 5)
            last_loss = None
            for _ in range(steps):
                last_loss = backprop_step(optimizer, batch_size=batch_size)

            # --- Eval / LR scheduler ---
            val_like_loss = estimate_val_like_loss()
            print(f"[TRAIN] Cycle {cycle} done. Last train_loss={last_loss}, "
                  f"val_like_loss={val_like_loss}")

            if val_like_loss is not None:
                scheduler.step(val_like_loss)
                if best_val_like_loss is None or val_like_loss < best_val_like_loss:
                    best_val_like_loss = val_like_loss
                    best_state = MODEL.state_dict()
                    torch.save(best_state, BEST_MODEL_PATH)
                    print(f"[TRAIN] New best model saved at {BEST_MODEL_PATH} "
                          f"(loss={val_like_loss:.6f})")

            # --- Periodic Elo ladder ---
            if total_selfplay_games % EVAL_LADDER_INTERVAL_GAMES == 0:
                run_stockfish_elo_ladder()

            # Save current model each cycle as a safety net
            torch.save(MODEL.state_dict(), MODEL_PATH)
            print(f"[TRAIN] Saved current model to {MODEL_PATH}")

    except KeyboardInterrupt:
        print("\n[TRAIN] Training interrupted by user (Ctrl-C). Saving final model...")

    finally:
        # Close Stockfish
        if sf_engine is not None:
            try:
                sf_engine.quit()
                print("[SF] Engine closed.")
            except Exception:
                pass

        # Save best / last model
        if best_state is not None:
            torch.save(best_state, MODEL_PATH)
            print(f"[TRAIN] Saved best-so-far model to {MODEL_PATH}.")
        elif MODEL is not None:
            torch.save(MODEL.state_dict(), MODEL_PATH)
            print("[TRAIN] Saved latest model.pt (no best model tracked).")


# ============================================================
#  RUN TRAINING IF EXECUTED DIRECTLY
# ============================================================

if __name__ == "__main__":
    if not TORCH_AVAILABLE or CnnValueNet is None:
        print("[TRAIN] Torch/CNN not available — cannot train.")
    else:
        # This will run until you hit Ctrl-C.
        train_forever(
            dataset_folder="datasets",
            max_dataset_positions=None,   # or an int if you want to cap
            batch_size=512,
            base_lr=1e-3,
            selfplay_games_per_cycle=10,
        )
