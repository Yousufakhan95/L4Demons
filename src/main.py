# ============================================================
#  CNN-BASED CHESSHACKS BOT + TRAINING ENGINE (DYNAMIC DEPTH)
#  - CNN value net over 8x8 planes
#  - Negamax + alpha-beta + quiescence + transposition table
#  - Dynamic search depth for actual games
#  - Pondering on opponent's time
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
import inspect
import shutil
import sys
import threading  # for pondering

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
MODEL_PATH = os.path.join(ROOT_DIR, "src","weights", "model.pt")
BEST_MODEL_PATH = os.path.join(ROOT_DIR,"src", "weights", "model_best.pt")

# ---------------------------
# Stockfish config
# ---------------------------
def find_stockfish():
    """Return a sensible Stockfish executable path for this environment.

    Order of preference:
      1. Bundled engine in `engines/` (try common names)
      2. System `stockfish` found via PATH
      3. Common Unix locations (/usr/bin, /usr/local/bin)
      4. Fallback to `engines/stockfish` (may be replaced by user)
    """
    engines_dir = os.path.join(ROOT_DIR, "engines")
    candidates = [
        os.path.join(engines_dir, "stockfish"),
        os.path.join(engines_dir, "stockfish.exe"),
        os.path.join(engines_dir, "stockfish-linux"),
        os.path.join(engines_dir, "stockfish_14"),
        os.path.join(engines_dir, "stockfish_15"),
    ]

    for p in candidates:
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p

    which_sf = shutil.which("stockfish")
    if which_sf:
        return which_sf

    for p in ["/usr/bin/stockfish", "/usr/local/bin/stockfish", "/root/project/engines/stockfish"]:
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p

    return os.path.join(engines_dir, "stockfish")


STOCKFISH_PATH = find_stockfish()
STOCKFISH_TIME_LIMIT = 0.03  # seconds per move for training/eval
STOCKFISH_MAX_MOVES = 200

# Elo ladder config
EVAL_ELOS = [1000]
EVAL_GAMES_PER_LEVEL = 4
EVAL_LADDER_INTERVAL_GAMES = 20  # run ladder every N self-play games

# ---------------------------
# Search config
# ---------------------------
# With no memory limit, we can afford deeper search.
BASE_SEARCH_DEPTH = 2          # starting point
MIN_SEARCH_DEPTH = 2           # clamp
MAX_SEARCH_DEPTH = 2           # clamp

SELFPLAY_DEPTH = 3             # depth in self-play
TRAIN_SEARCH_DEPTH = 4         # depth vs Stockfish in training

# ---------------------------
# Data config
# ---------------------------
MIN_FULLMOVES_FOR_PGN_LABEL = 10  # start labeling after move 10

# ---------------------------
# Elo weighting config
# ---------------------------
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
#  CNN VALUE NETWORK (BEEFED UP)
# ============================================================

if TORCH_AVAILABLE and nn is not None:

    class CnnValueNet(nn.Module):
        """
        CNN over board planes + dense head with extra meta features.
        Input: flat vector length FEATURE_DIM
        Output: scalar value in [-1, 1] from White's POV for given board.
        """
        def __init__(self, input_dim: int, num_planes: int = NUM_PLANES):
            super().__init__()
            self.num_planes = num_planes
            board_vec_dim = num_planes * BOARD_SIZE * BOARD_SIZE
            self.board_vec_dim = board_vec_dim
            self.meta_dim = input_dim - board_vec_dim  # should be 4

            # Wider convolutional trunk
            self.conv1 = nn.Conv2d(num_planes, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

            conv_output_dim = 128 * BOARD_SIZE * BOARD_SIZE  # 128 channels * 8 * 8

            self.fc1 = nn.Linear(conv_output_dim + self.meta_dim, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.fc_out = nn.Linear(512, 1)
            self.dropout = nn.Dropout(0.25)

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
    return total_cp / 100.0


def get_dynamic_depth(board: chess.Board) -> int:
    """
    Dynamic depth for actual dashboard games.
    """
    depth = BASE_SEARCH_DEPTH
    return depth


# ============================================================
#  NEGAMAX + ALPHA-BETA + QUIESCENCE + TT
# ============================================================

# Simple transposition table:
# key -> (depth, score, flag)
# flag ∈ {"EXACT", "LOWER", "UPPER"}
TT = {}
TT_HITS = 0
TT_MISSES = 0


def softmax(scores: List[float]) -> List[float]:
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(scores)] * len(scores)
    return [e / total for e in exps]


def quiescence(board: chess.Board, alpha: float, beta: float, depth_cap: int = 6) -> float:
    """
    Quiescence search:
      - Only explore capture moves (and promotions)
      - Slightly deeper now since we aren't memory-bound.
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
    Negamax with alpha-beta pruning, quiescence, and a transposition table.
    Scores are always from POV of side to move.
    """
    global TT_HITS, TT_MISSES

    if depth == 0 or board.is_game_over():
        return quiescence(board, alpha, beta)

    key = None
    if hasattr(board, "transposition_key"):
        try:
            key = board.transposition_key()
        except Exception:
            key = None

    orig_alpha = alpha

    if key is not None:
        entry = TT.get(key)
        if entry is not None:
            TT_HITS += 1
            entry_depth, entry_score, entry_flag = entry
            if entry_depth >= depth:
                if entry_flag == "EXACT":
                    return entry_score
                elif entry_flag == "LOWER":
                    alpha = max(alpha, entry_score)
                elif entry_flag == "UPPER":
                    beta = min(beta, entry_score)
                if alpha >= beta:
                    return entry_score
        else:
            TT_MISSES += 1

    max_eval = -float("inf")
    legal_moves = list(board.generate_legal_moves())

    # Simple move ordering: captures first
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

    # Store in TT
    if key is not None:
        if max_eval <= orig_alpha:
            flag = "UPPER"
        elif max_eval >= beta:
            flag = "LOWER"
        else:
            flag = "EXACT"
        TT[key] = (depth, max_eval, flag)

    return max_eval


def choose_move_with_search(
    board: chess.Board,
    legal_moves,
    depth: Optional[int] = None,
):
    """
    Use negamax + alpha-beta to pick the best move.
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
#  PONDERING (THINK ON OPPONENT'S TIME)
# ============================================================

# Cache: board_fen -> best_move we computed while pondering.
PONDER_CACHE = {}
PONDER_LOCK = threading.Lock()
PONDER_THREAD: Optional[threading.Thread] = None
PONDER_STOP = threading.Event()


def _board_key(board: chess.Board) -> str:
    return board.fen()


def _ponder_set_result(board_fen: str, move: Move):
    with PONDER_LOCK:
        # You can grow this more if you want; it's just RAM now.
        if len(PONDER_CACHE) > 4096:
            PONDER_CACHE.clear()
        PONDER_CACHE[board_fen] = move


def _ponder_get_move(board: chess.Board) -> Optional[Move]:
    key = _board_key(board)
    with PONDER_LOCK:
        return PONDER_CACHE.get(key)


def stop_pondering():
    """Signal current ponder thread to stop (if any)."""
    global PONDER_THREAD
    PONDER_STOP.set()
    if PONDER_THREAD is not None and PONDER_THREAD.is_alive():
        try:
            PONDER_THREAD.join(timeout=0.02)
        except Exception:
            pass
    PONDER_THREAD = None
    PONDER_STOP.clear()


def _ponder_worker(root_fen: str):
    """
    Background worker:
      - root_fen: position AFTER our move (opponent to move).
      - For each opponent reply, compute our best move in that line and cache it.
    """
    try:
        root = chess.Board(root_fen)
    except Exception as e:
        print("[PONDER] Failed to init board from fen:", e)
        return

    opp_moves = list(root.generate_legal_moves())
    for opp_move in opp_moves:
        if PONDER_STOP.is_set():
            break

        child = root.copy()
        child.push(opp_move)  # now it's our turn

        legal = list(child.generate_legal_moves())
        if not legal:
            continue

        # Slightly DEEPER search while pondering (if possible).
        try:
            ponder_depth = min(MAX_SEARCH_DEPTH + 1, 6)
            reply, _ = choose_move_with_search(child, legal_moves=legal, depth=ponder_depth)
        except Exception as e:
            print("[PONDER] Error during ponder search:", e)
            continue

        if reply is None:
            continue

        _ponder_set_result(child.fen(), reply)


def start_pondering_from(board_after_our_move: chess.Board):
    """
    Start a background ponder thread from 'board_after_our_move', which
    should be the position RIGHT AFTER we make our move (opponent to move).
    """
    if not TORCH_AVAILABLE or MODEL is None:
        return

    stop_pondering()
    fen = board_after_our_move.fen()

    global PONDER_THREAD
    PONDER_THREAD = threading.Thread(
        target=_ponder_worker,
        args=(fen,),
        daemon=True,
    )
    PONDER_STOP.clear()
    PONDER_THREAD.start()


# ============================================================
#  CHESSHACKS ENTRYPOINTS (ACTUAL GAMES)
# ============================================================

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    """
    Called for every move request by ChessHacks.

    - Uses dynamic depth for search (get_dynamic_depth).
    - Depth scales with phase of game + checks.
    - Uses pondering results if available.
    """
    # Stop any previous pondering (but keep cache).
    stop_pondering()

    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available.")

    # 1) If we already pondered this exact position, use that.
    cached_move = _ponder_get_move(ctx.board)
    if cached_move is not None and cached_move in legal_moves:
        probs = {m: (1.0 if m == cached_move else 0.0) for m in legal_moves}
        ctx.logProbabilities(probs)
        chosen_move = cached_move
    else:
        # 2) Normal search.
        chosen_move, prob_map = choose_move_with_search(ctx.board, legal_moves, depth=None)
        ctx.logProbabilities(prob_map)

    # 3) After deciding on our move, start pondering from the
    #    new position (opponent to move) in a background thread.
    board_after = ctx.board.copy()
    board_after.push(chosen_move)
    start_pondering_from(board_after)

    return chosen_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    """
    Called when a new game starts.
    """
    stop_pondering()
    with PONDER_LOCK:
        PONDER_CACHE.clear()


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
    - Above MAX → 1.0
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
            winner = row.get("winner", "").strip().lower()
            if winner == "white":
                winner_color = chess.WHITE
            elif winner == "black":
                winner_color = chess.BLACK
            else:
                winner_color = None

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

    all_positions: List[Tuple] = []

    # Parallel PGN loading
    if pgn_files:
        total = len(pgn_files)
        print(f"[DATA] Parallel loading {total} PGN files...")
        completed = 0

        if max_positions is not None:
            per_file_limit = max(1, max_positions // max(total, 1))
        else:
            per_file_limit = None

        max_workers = min(8, max(1, os.cpu_count() or 1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for f_name in pgn_files:
                path = os.path.join(folder, f_name)
                futures[ex.submit(load_pgn_dataset, path, per_file_limit)] = f_name

            pending = set(futures.keys())
            try:
                idle_rounds = 0
                while pending:
                    done, pending = concurrent.futures.wait(
                        pending,
                        timeout=60,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    if not done:
                        idle_rounds += 1
                        print("[DATA] Warning: no PGN finished within 60s; checking pending tasks...")
                        if idle_rounds >= 3:
                            stalled = [futures.get(fut, '<unknown>') for fut in pending]
                            print(f"[DATA] Warning: parallel loading appears stalled. Stalled files: {stalled}")
                            break
                        continue

                    idle_rounds = 0
                    for fut in list(done):
                        fname = futures.get(fut, "<unknown>")
                        try:
                            pos = fut.result(timeout=5)
                            all_positions.extend(pos)
                            print(f"[DATA] PGN {fname}: {len(pos)} positions")
                        except Exception as e:
                            print(f"[DATA] Failed to load PGN {fname}: {e}")

                        completed += 1
                        print(f"[DATA] PGN progress: {completed}/{total}")

                        if max_positions is not None and len(all_positions) >= max_positions:
                            all_positions = all_positions[:max_positions]
                            print(f"[DATA] Reached max_positions={max_positions} from PGNs.")
                            pending.clear()
                            break

                if pending:
                    print(f"[DATA] Falling back to sequential load for {len(pending)} pending PGN(s).")
                    for fut in list(pending):
                        fname = futures.get(fut, "<unknown>")
                        try:
                            path = os.path.join(folder, fname)
                        except Exception:
                            path = None

                        try:
                            fut.cancel()
                        except Exception:
                            pass

                        if path and os.path.exists(path):
                            try:
                                pos = load_pgn_dataset(path, per_file_limit)
                                all_positions.extend(pos)
                                print(f"[DATA] (sequential) PGN {fname}: {len(pos)} positions")
                            except Exception as e:
                                print(f"[DATA] (sequential) Failed to load PGN {fname}: {e}")
                        else:
                            print(f"[DATA] Could not determine path for stalled PGN {fname}; skipping.")

                        completed += 1
                        print(f"[DATA] PGN progress: {completed}/{total}")

                        if max_positions is not None and len(all_positions) >= max_positions:
                            all_positions = all_positions[:max_positions]
                            print(f"[DATA] Reached max_positions={max_positions} from PGNs.")
                            break
            except KeyboardInterrupt:
                print("[DATA] Loading interrupted by user.")
            except Exception as e:
                print(f"[DATA] Unexpected error during parallel PGN loading: {e}")
                for fut in list(pending):
                    try:
                        fname = futures.get(fut, "<unknown>")
                        pos = fut.result(timeout=1)
                        all_positions.extend(pos)
                        print(f"[DATA] PGN {fname}: {len(pos)} positions (late)")
                    except Exception:
                        pass
                pending.clear()

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
        def __init__(self, cap: int = 1_000_000):
            self.cap = cap
            # store (x, y, weight)
            self.data = []

        def add(self, x, y, weight: float = 1.0):
            if len(self.data) >= self.cap:
                self.data.pop(0)
            self.data.append((x, float(y), float(weight)))

        def extend(self, items):
            for item in items:
                if len(item) == 3:
                    x, y, w = item
                else:
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


def self_play_game(max_moves: int = 600, depth: int = SELFPLAY_DEPTH):
    """
    Self-play game with the current MODEL.
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
    try:
        plies = len(board.move_stack)
        fullmoves = plies // 2
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            term = getattr(outcome, "termination", None)
            winner = getattr(outcome, "winner", None)
            print(f"[SELFPLAY] Finished: plies={plies}, fullmoves={fullmoves}, termination={term}, winner={winner}")
        else:
            print(f"[SELFPLAY] Finished: plies={plies}, fullmoves={fullmoves}, result={res}")
    except Exception as _e:
        print(f"[SELFPLAY] Finished: could not compute outcome details: {_e}")
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
    Launch Stockfish with optional strength limiting.
    """
    if not os.path.exists(STOCKFISH_PATH):
        print(f"[SF] Stockfish not found at: {STOCKFISH_PATH}")
        return None, False

    try:
        engine = chess.engine.SimpleEngine.popen_uci([STOCKFISH_PATH])
        print("[SF] Stockfish engine started.")
    except Exception as e:
        print(f"[SF] Error launching Stockfish at {STOCKFISH_PATH}: {e}")
        try:
            alt = shutil.which("stockfish")
            if alt and alt != STOCKFISH_PATH:
                engine = chess.engine.SimpleEngine.popen_uci([alt])
                print(f"[SF] Stockfish engine started via PATH at {alt}.")
                return engine, True
        except Exception as e2:
            print(f"[SF] Fallback via PATH failed: {e2}")

        alt2 = os.path.join(os.path.dirname(STOCKFISH_PATH), "stockfish")
        if os.path.exists(alt2) and os.access(alt2, os.X_OK):
            try:
                engine = chess.engine.SimpleEngine.popen_uci([alt2])
                print(f"[SF] Stockfish engine started via bundled path {alt2}.")
                return engine, True
            except Exception as e3:
                print(f"[SF] Fallback via bundled path failed: {e3}")

        return None, False

    try:
        opts = engine.options
    except Exception as e:
        print(f"[SF] Could not read engine options: {e}")
        return engine, True

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

    if "Skill Level" in opts:
        if sf_skill is None:
            sf_skill = 0
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
    """
    if MODEL is None or REPLAY is None or len(REPLAY) == 0:
        return None

    MODEL.train()
    xs, ys, ws = REPLAY.sample(batch_size)
    preds = MODEL(xs)
    loss_vec = (preds - ys) ** 2

    ws = ws / (ws.mean() + 1e-8)
    loss = (ws * loss_vec).mean()

    optimizer.zero_grad()
    loss.backward()
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
    try:
        sig = inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau)
        if "verbose" in sig.parameters:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5, verbose=True
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
    except Exception:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

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
            loss = F.mse_loss(preds, ys)
        return float(loss.item())

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
                for i in range(2):
                    color = chess.WHITE if i % 2 == 0 else chess.BLACK
                    pos_sf, res = game_vs_stockfish(sf_engine, our_color=color)
                    if REPLAY is not None:
                        REPLAY.extend(pos_sf)
                    print(f"[TRAIN] vs Stockfish ({'White' if color == chess.WHITE else 'Black'}): "
                          f"{len(pos_sf)} positions, result={res}, replay={len(REPLAY)}")

            # --- Backprop steps ---
            steps = max(20, selfplay_games_per_cycle * 8)
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

            if total_selfplay_games % EVAL_LADDER_INTERVAL_GAMES == 0:
                run_stockfish_elo_ladder()

            torch.save(MODEL.state_dict(), MODEL_PATH)
            print(f"[TRAIN] Saved current model to {MODEL_PATH}")

    except KeyboardInterrupt:
        print("\n[TRAIN] Training interrupted by user (Ctrl-C). Saving final model...")

    finally:
        if sf_engine is not None:
            try:
                sf_engine.quit()
                print("[SF] Engine closed.")
            except Exception:
                pass

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
        train_forever(
            dataset_folder="datasets",
            max_dataset_positions=None,
            batch_size=512,
            base_lr=1e-3,
            selfplay_games_per_cycle=10,
        )
