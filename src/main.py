# ============================================================
#  FULL CHESSHACKS BOT + TRAINING + UNIVERSAL DATA LOADER
#  (GPU, negamax+alpha-beta, parallel PGN loading, replay buf,
#   upgraded net, validation, LR scheduler, best-model saving)
# ============================================================

from src.utils import chess_manager, GameContext
from chess import Board, Move
import chess
import chess.engine
import random
import math
import os
import csv
import numpy as np
from typing import Optional, List, Tuple
import concurrent.futures
import platform
import time

# ============================================================
#  TRY TORCH — BUT BOT MUST WORK WITHOUT IT
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
#  CONSTANTS + ROOT
# ============================================================

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
BOARD_SIZE = 8
NUM_PIECE_PLANES = 12   # 6 piece types per color
NUM_EXTRA_PLANES = 1    # side-to-move plane
FEATURE_DIM = BOARD_SIZE * BOARD_SIZE * (NUM_PIECE_PLANES + NUM_EXTRA_PLANES) + 4
# 4 castling flags (white/black castle rights)

MODEL = None
DEVICE = "cuda"
MODEL_PATH = os.path.join(ROOT_DIR, "model.pt")
BEST_MODEL_PATH = os.path.join(ROOT_DIR, "model_best.pt")

# Search config
SEARCH_DEPTH = 2          # main engine depth for inference (root)
TRAIN_SEARCH_DEPTH = 3    # search depth to use during training vs Stockfish
SELFPLAY_DEPTH = 1        # (deprecated) depth used during self-play (keep small for speed)

# Data config
MIN_FULLMOVES_FOR_PGN_LABEL = 10  # only label positions after move 10 to reduce early-game noise

# Stockfish config
STOCKFISH_PATH = None  # Will be auto-detected or set manually
STOCKFISH_TIME_LIMIT = 0.1  # Time per move in seconds
STOCKFISH_SKILL_LEVEL = 20  # 0-20, where 20 is strongest
STOCKFISH_ELO = 1300.0  # Starting Elo; will increase if model wins consistently
STOCKFISH_ELO_MAX = 3000.0  # Max Elo to prevent unlimited growth

# Model rating tracking (simple Elo tracker updated after each training game)
MODEL_ELO = 1500.0
ELO_K = 20.0


# ============================================================
#  FEATURE ENCODING
# ============================================================

def piece_plane_index(piece: chess.Piece) -> int:
    # 0–5 white Pawn..King, 6–11 black Pawn..King
    return (0 if piece.color else 6) + (piece.piece_type - 1)


def board_to_tensor(board: chess.Board):
    """
    Encode board into a 1D tensor of size FEATURE_DIM.
    Returns None if torch is not available.
    """
    if not TORCH_AVAILABLE or torch is None:
        return None

    planes = torch.zeros(NUM_PIECE_PLANES + NUM_EXTRA_PLANES, 8, 8, dtype=torch.float32)

    for square, piece in board.piece_map().items():
        rank = square // 8
        file = square % 8
        planes[piece_plane_index(piece), rank, file] = 1.0

    # side to move plane
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
#  MODEL
# ============================================================

if TORCH_AVAILABLE and nn is not None:

    class ResidualBlock(nn.Module):
        """
        Simple residual MLP block: x -> LN -> GELU -> Linear -> Dropout -> +x
        No hand-crafted eval, just deeper non-linear function approximator.
        """
        def __init__(self, dim, dropout=0.2):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.fc = nn.Linear(dim, dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            residual = x
            x = self.norm(x)
            x = F.gelu(x)
            x = self.fc(x)
            x = self.dropout(x)
            return x + residual


    class SimpleValueNet(nn.Module):
        """
        Upgraded value net:
          - Wider hidden layer
          - Stack of residual blocks with LayerNorm + GELU
          - Final projection to scalar in [-1, 1]
        Input = flat FEATURE_DIM, output = scalar value.
        """
        def __init__(self, input_dim):
            super().__init__()
            hidden_dim = 1024
            num_blocks = 4

            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.blocks = nn.ModuleList(
                [ResidualBlock(hidden_dim, dropout=0.2) for _ in range(num_blocks)]
            )

            self.head_norm = nn.LayerNorm(hidden_dim)
            self.head_fc1 = nn.Linear(hidden_dim, 256)
            self.head_dropout = nn.Dropout(0.2)
            self.head_fc2 = nn.Linear(256, 1)

        def forward(self, x):
            x = self.input_proj(x)
            for block in self.blocks:
                x = block(x)

            x = self.head_norm(x)
            x = F.gelu(x)
            x = self.head_fc1(x)
            x = F.gelu(x)
            x = self.head_dropout(x)
            x = self.head_fc2(x)
            x = torch.tanh(x).squeeze(-1)   # [-1, 1]
            return x

else:
    SimpleValueNet = None


def init_model():
    """
    Init MODEL (from model.pt if present).
    """
    global MODEL, DEVICE
    if not TORCH_AVAILABLE or SimpleValueNet is None:
        MODEL = None
        print("[ML] Torch not available, running without ML.")
        return

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ML] Using device: {DEVICE}")
    MODEL = SimpleValueNet(FEATURE_DIM).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        try:
            state = torch.load(MODEL_PATH, map_location=DEVICE)
            if isinstance(state, dict) and "model_state_dict" in state:
                # in case of older save formats
                state = state["model_state_dict"]
            MODEL.load_state_dict(state)
            print(f"[ML] Loaded model from {MODEL_PATH}")
        except Exception as e:
            print(f"[ML] Failed to load model.pt: {e}")
    else:
        print("[ML] No model.pt found — using fresh weights.")

    MODEL.eval()


init_model()


# ============================================================
#  MOVE SELECTION HELPERS
# ============================================================

def softmax(scores):
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(scores)] * len(scores)
    return [e / total for e in exps]

# ============================================================
#  DYNAMIC PIECE EVALUATION (POSITION-SENSITIVE)
# ============================================================

def get_game_phase(board: chess.Board) -> float:
    """
    Returns game phase: 0.0 = opening/middlegame, 1.0 = endgame
    Based on remaining material.
    """
    # Count material (excluding pawns for phase calculation)
    material = 0
    piece_values = {chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    
    for piece_type, value in piece_values.items():
        material += len(board.pieces(piece_type, chess.WHITE)) * value
        material += len(board.pieces(piece_type, chess.BLACK)) * value
    
    # Max material = 2*(9+2*5+2*3+2*3) = 62
    # Endgame roughly when material < 20
    return max(0.0, min(1.0, 1.0 - (material / 30.0)))


def piece_square_bonus(piece: chess.Piece, square: int, game_phase: float) -> float:
    """
    Position-based bonuses for pieces. Returns bonus in centipawns.
    """
    rank = square // 8
    file = square % 8
    
    # Center squares are generally good
    center_dist = abs(3.5 - rank) + abs(3.5 - file)
    centralization = (7 - center_dist) / 7.0
    
    # Adjust rank for black pieces (flip perspective)
    if not piece.color:
        rank = 7 - rank
    
    bonus = 0.0
    
    if piece.piece_type == chess.PAWN:
        # Pawns: advance bonus, center files better
        advance_bonus = rank * 10  # up to 70 for 7th rank
        center_file_bonus = 10 * (1 - abs(3.5 - file) / 3.5)
        
        # Passed pawn detection (simplified)
        if rank >= 4:  # advanced pawn
            bonus += 20 + (rank - 4) * 15
        
        bonus += advance_bonus + center_file_bonus
        
    elif piece.piece_type == chess.KNIGHT:
        # Knights: love center, hate edges
        edge_penalty = 0
        if rank == 0 or rank == 7:
            edge_penalty += 20
        if file == 0 or file == 7:
            edge_penalty += 20
        
        # Outpost bonus (advanced + protected)
        outpost_bonus = 0
        if rank >= 4 and rank <= 6:
            outpost_bonus = 25
            
        bonus += centralization * 30 - edge_penalty + outpost_bonus
        
    elif piece.piece_type == chess.BISHOP:
        # Bishops: like long diagonals, mobility
        diagonal_bonus = 0
        if (rank == file) or (rank + file == 7):  # main diagonals
            diagonal_bonus = 15
            
        bonus += centralization * 15 + diagonal_bonus
        
    elif piece.piece_type == chess.ROOK:
        # Rooks: 7th rank, open files
        if rank == 6:  # 7th rank
            bonus += 40
        elif rank == 7:  # 8th rank
            bonus += 20
            
        # Prefer central files in endgame
        if game_phase > 0.5:
            bonus += (1 - abs(3.5 - file) / 3.5) * 20
            
    elif piece.piece_type == chess.QUEEN:
        # Queen: activity bonus, but not too early
        if game_phase < 0.3:  # opening/early middlegame
            # Penalize early development
            if rank > 1:
                bonus -= 20
        else:
            # Likes centralization but not as much as minor pieces
            bonus += centralization * 10
            
    elif piece.piece_type == chess.KING:
        # King: safety in opening/middlegame, activity in endgame
        if game_phase < 0.7:
            # Prefer castled position
            if (piece.color and rank == 0) or (not piece.color and rank == 7):
                if file in [0, 1, 6, 7]:  # likely castled
                    bonus += 40
                else:  # center = dangerous
                    bonus -= 30
        else:
            # Endgame: king activity is key
            bonus += centralization * 30
    
    return bonus


def dynamic_material_eval(board: chess.Board) -> float:
    """
    Advanced material evaluation with position-dependent piece values.
    Returns score from White's POV in roughly [-20, 20] range.
    """
    # Base piece values
    base_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0,  # King has no material value
    }
    
    game_phase = get_game_phase(board)
    
    # Adjust base values by game phase
    phase_adjusted_values = base_values.copy()
    
    # Knights slightly better in closed positions (approximated by pawn count)
    pawn_count = len(board.pieces(chess.PAWN, chess.WHITE)) + len(board.pieces(chess.PAWN, chess.BLACK))
    if pawn_count > 12:  # closed position
        phase_adjusted_values[chess.KNIGHT] += 10
        phase_adjusted_values[chess.BISHOP] -= 10
    
    # Rooks stronger in endgame
    phase_adjusted_values[chess.ROOK] += int(50 * game_phase)
    
    # Bishop pair bonus
    white_bishops = len(board.pieces(chess.BISHOP, chess.WHITE))
    black_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
    bishop_pair_bonus = 50 if white_bishops >= 2 else 0
    bishop_pair_bonus -= 50 if black_bishops >= 2 else 0
    
    score = bishop_pair_bonus
    
    # Calculate position-adjusted values
    for square, piece in board.piece_map().items():
        base_val = phase_adjusted_values[piece.piece_type]
        pos_bonus = piece_square_bonus(piece, square, game_phase)
        
        total_val = base_val + pos_bonus
        
        if piece.color == chess.WHITE:
            score += total_val
        else:
            score -= total_val
    
    # Mobility bonus (simplified - just count legal moves)
    if not board.is_game_over():
        white_mobility = len(list(board.legal_moves)) if board.turn else 0
        board.push(chess.Move.null())  # switch turn
        black_mobility = len(list(board.legal_moves)) if not board.turn else 0
        board.pop()
        
        # Each extra move worth ~5 centipawns
        score += (white_mobility - black_mobility) * 5
    
    # Scale to roughly [-20, 20] -> [-1, 1]
    return score / 2000.0


# Keep simple material as fallback
def basic_material(board: chess.Board) -> float:
    """
    Very simple material count from White's POV.
    Returns a score in roughly [-10, 10], which we then scale down.
    """
    values = {
        chess.PAWN: 100,
        chess.KNIGHT: 300,
        chess.BISHOP: 300,
        chess.ROOK: 500,
        chess.QUEEN: 900,
    }

    score = 0
    for piece_type, v in values.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * v
        score -= len(board.pieces(piece_type, chess.BLACK)) * v

    # Scale down to something like [-10,10] -> [-1,1]
    return score / 1000.0
# ============================================================

def evaluate_position(board: chess.Board) -> float:
    """
    Evaluate board from POV of side to move.

    - Primary: neural net value in [-1, 1]
    - Safety net: dynamic material eval with position-sensitive weights
    """
    # If no NN, fall back to dynamic material evaluation
    if not TORCH_AVAILABLE or MODEL is None:
        # Dynamic material from side-to-move POV
        mat = dynamic_material_eval(board)
        return mat if board.turn == chess.WHITE else -mat

    x = board_to_tensor(board)
    if x is None:
        mat = dynamic_material_eval(board)
        return mat if board.turn == chess.WHITE else -mat

    # NN value
    with torch.no_grad():
        x = x.to(DEVICE).unsqueeze(0)
        nn_val = MODEL(x)[0].item()  # already from side-to-move POV

    # Dynamic material value from side-to-move POV
    mat_val = dynamic_material_eval(board)
    if not board.turn:
        mat_val = -mat_val

    # Blend them: NN mostly in charge, dynamic eval prevents stupid hangs
    # Adjust blend based on game phase - material more important in endgame
    game_phase = get_game_phase(board)
    alpha = 0.85 - (0.1 * game_phase)  # NN weight: 0.85 in opening, 0.75 in endgame
    beta = 0.15 + (0.1 * game_phase)   # Material weight: 0.15 in opening, 0.25 in endgame

    blended = alpha * nn_val + beta * mat_val
    return float(blended)


# ============================================================
#  NEGAMAX + ALPHA-BETA SEARCH
# ============================================================

def negamax(board: chess.Board, depth: int, alpha: float, beta: float) -> float:
    """
    Negamax search with alpha-beta pruning.
    Assumes evaluate_position(board) returns value from POV of side to move.
    """
    # Base case: leaf node or game over
    if depth == 0 or board.is_game_over():
        return evaluate_position(board)

    max_eval = -float("inf")

    for move in board.generate_legal_moves():
        board.push(move)
        # Opponent to move in child, so negate value
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
    If depth is None, uses global SEARCH_DEPTH.
    """
    if not TORCH_AVAILABLE or MODEL is None:
        n = len(legal_moves)
        return random.choice(legal_moves), {m: 1.0 / n for m in legal_moves}

    if depth is None:
        depth = SEARCH_DEPTH

    best_score = -float("inf")
    best_move = None
    scores = []

    for move in legal_moves:
        board.push(move)
        score = -negamax(board, depth - 1, -float("inf"), float("inf"))
        board.pop()

        scores.append(score)
        if score > best_score or best_move is None:
            best_score = score
            best_move = move

    probs = softmax(scores)
    prob_map = {m: p for m, p in zip(legal_moves, probs)}
    return best_move, prob_map


def choose_move_ml(board: chess.Board, legal_moves, temperature: Optional[float] = 1.0):
    """
    Use NN eval to pick a move via 1-ply evaluation.
    - If temperature is None or <= 0 -> pure greedy (argmax on NN score).
    - If temperature > 0 -> sample from softmax(scores / temperature).
    No hand-crafted eval; uses only evaluate_position() on legal moves.
    """
    if not TORCH_AVAILABLE or MODEL is None:
        n = len(legal_moves)
        return random.choice(legal_moves), {m: 1.0 / n for m in legal_moves}

    if not legal_moves:
        return None, {}

    scores = []
    for move in legal_moves:
        board.push(move)
        s = evaluate_position(board)
        board.pop()
        scores.append(s)

    # Greedy / deterministic mode: debug true NN strength
    if temperature is None or temperature <= 0:
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_move = legal_moves[best_idx]
        prob_map = {m: 0.0 for m in legal_moves}
        prob_map[best_move] = 1.0
        return best_move, prob_map

    if temperature != 1.0:
        scores = [s / temperature for s in scores]

    probs = softmax(scores)
    move = random.choices(legal_moves, weights=probs, k=1)[0]
    prob_map = {m: p for m, p in zip(legal_moves, probs)}
    return move, prob_map


# ============================================================
#  CHESSHACKS ENTRYPOINTS
# ============================================================

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    """
    Called for every move request by ChessHacks.
    Uses negamax + alpha-beta for stronger play (NN eval only).
    """
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available.")

    move, prob_map = choose_move_with_search(ctx.board, legal_moves, depth=SEARCH_DEPTH)
    ctx.logProbabilities(prob_map)
    return move


@chess_manager.reset
def reset_func(ctx: GameContext):
    """
    Called when a new game starts.
    """
    pass


# ============================================================
#  RESULT → VALUE HELPERS
# ============================================================

def game_result_to_value(result: str) -> float:
    """
    Convert PGN result string to value from White's POV:
      "1-0"   -> +1
      "0-1"   -> -1
      "1/2-1/2" or anything else -> 0
    """
    if result == "1-0":
        return 1.0
    elif result == "0-1":
        return -1.0
    else:
        return 0.0


# ============================================================
#  PGN DATA LOADER
# ============================================================

def load_pgn_dataset(path, max_positions=None):
    """
    Load PGN file: for each game, assign a value from the POV of
    the *side to move* in each position.

    Rules:
      - If the game is a decisive result:
          value = +1.0  if side-to-move eventually wins
                  -1.0  if side-to-move eventually loses
      - If the game is a draw or has no standard result:
          value = 0.0
      - To reduce noisy opening labels, we only start collecting
        after MIN_FULLMOVES_FOR_PGN_LABEL.
    """
    if not TORCH_AVAILABLE or torch is None:
        return []

    import chess.pgn

    basename = os.path.basename(path)
    print(f"[DATA] Starting PGN load: {basename}")
    positions = []
    game_count = 0

    with open(path, "r", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            game_count += 1

            result = game.headers.get("Result", "*")
            if result == "1-0":
                winner_color = chess.WHITE
            elif result == "0-1":
                winner_color = chess.BLACK
            else:
                winner_color = None  # treat as draw/unknown -> 0.0

            board = game.board()
            for mv in game.mainline_moves():
                board.push(mv)

                # Skip very early opening moves (result mostly uncorrelated)
                if board.fullmove_number < MIN_FULLMOVES_FOR_PGN_LABEL:
                    continue

                x = board_to_tensor(board)
                if x is None:
                    continue

                # Value from POV of *current* side to move
                if winner_color is None:
                    val = 0.0
                else:
                    # If side to move is the eventual winner -> +1, else -1
                    val = 1.0 if board.turn == winner_color else -1.0

                positions.append((x, val))

                if max_positions and len(positions) >= max_positions:
                    print(
                        f"[DATA] Finished PGN {basename}: "
                        f"{game_count} games, {len(positions)} positions (truncated)."
                    )
                    return positions

    print(f"[DATA] Finished PGN {basename}: {game_count} games, {len(positions)} positions.")
    return positions



# ============================================================
#  LICHESS GAMES CSV LOADER
# ============================================================

def load_lichess_games_csv(path, max_positions=None):
    """
    Load a Lichess games CSV like:
      id,rated,created_at,...,winner,...,moves,...

    For each game:
      - Determine winner_color from 'winner' column.
      - Replay moves using SAN.
      - After each move (optionally skipping early moves), store
        (features, value) where:

          value = +1.0 if side-to-move eventually wins
                  -1.0 if side-to-move eventually loses
                   0.0 for draws/unknown
    """
    if not TORCH_AVAILABLE or torch is None:
        print("[DATA] Torch not available; cannot load Lichess CSV.")
        return []

    positions = []
    basename = os.path.basename(path)
    print(f"[DATA] Starting Lichess CSV load: {basename}")

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            winner = row.get("winner", "").strip().lower()

            if winner == "white":
                winner_color = chess.WHITE
            elif winner == "black":
                winner_color = chess.BLACK
            else:
                winner_color = None  # draw/unknown

            moves_str = row.get("moves", "")
            if not moves_str:
                continue

            board = chess.Board()
            moves = moves_str.split()

            for mv in moves:
                try:
                    board.push_san(mv)
                except ValueError:
                    # Invalid SAN -> bail out of this game
                    break

                if board.fullmove_number < MIN_FULLMOVES_FOR_PGN_LABEL:
                    continue

                x = board_to_tensor(board)
                if x is None:
                    continue

                if winner_color is None:
                    val = 0.0
                else:
                    val = 1.0 if board.turn == winner_color else -1.0

                positions.append((x, val))

                if max_positions and len(positions) >= max_positions:
                    print(
                        f"[DATA] Lichess CSV {basename} -> "
                        f"{len(positions)} positions (truncated)."
                    )
                    return positions

    print(f"[DATA] Lichess CSV {basename} -> {len(positions)} positions.")
    return positions



# ============================================================
#  UNIVERSAL DATASET LOADER (LOAD EVERYTHING IN datasets/)
#  + CACHING
# ============================================================

def load_all_datasets(folder="datasets", max_positions=None):
    """
    Loads all datasets in a folder:
      - .pgn     -> PGN games (loaded in parallel per file)
      - .pt/.pth -> torch-saved tensors
      - .npz     -> x,y arrays
      - .csv     -> either Lichess games CSV or generic FEN/features CSV

    Returns a list of (features_tensor, value).
    """
    if not os.path.isabs(folder):
        folder = os.path.join(ROOT_DIR, folder)

    if not os.path.exists(folder):
        print(f"[DATA] No dataset folder: {folder}")
        return []

    # ---------- CACHE CHECK ----------
    cache_path = os.path.join(folder, "cached_dataset.pt")
    if TORCH_AVAILABLE and os.path.exists(cache_path):
        print(f"[DATA] Loading cached dataset from {cache_path}")
        try:
            data = torch.load(cache_path, map_location="cpu")
            xs = torch.tensor(data["x"], dtype=torch.float32)
            ys = torch.tensor(data["y"], dtype=torch.float32)
            positions = list(zip(xs, ys))
            if max_positions is not None and len(positions) > max_positions:
                positions = positions[:max_positions]
            print(f"[DATA] Loaded {len(positions)} cached positions.")
            return positions
        except Exception as e:
            print(f"[DATA] Failed to load cached dataset: {e}")
    # -------------------------------

    files = os.listdir(folder)
    all_pos = []

    print(f"[DATA] Files in {folder}: {files}")

    # ---------- PARALLEL PGN LOADING WITH PROGRESS ----------
    pgn_files = [f for f in files if f.lower().endswith(".pgn")]
    other_files = [f for f in files if f not in pgn_files]

    if pgn_files:
        print(f"[DATA] Loading {len(pgn_files)} PGN files in parallel...")

        total_pgn = len(pgn_files)
        completed = 0

        if max_positions is not None:
            per_file_limit = max(1, max_positions // total_pgn)
        else:
            per_file_limit = None

        with concurrent.futures.ThreadPoolExecutor() as ex:
            futures = {}
            for file in pgn_files:
                path = os.path.join(folder, file)
                print(f"[DATA] Scheduling PGN: {file}")
                futures[ex.submit(load_pgn_dataset, path, per_file_limit)] = file

            for fut in concurrent.futures.as_completed(futures):
                file = futures[fut]
                try:
                    positions = fut.result()
                    all_pos.extend(positions)
                    print(f"[DATA] PGN {file} -> {len(positions)} positions")
                except Exception as e:
                    print(f"[DATA] Failed to load PGN {file} in parallel: {e}")

                completed += 1
                progress = (completed / total_pgn) * 100.0
                print(f"[DATA] PGN progress: {progress:.1f}% ({completed}/{total_pgn})")

                if max_positions is not None and len(all_pos) >= max_positions:
                    all_pos = all_pos[:max_positions]
                    print(f"[DATA] Reached max_positions={max_positions} from PGNs.")
                    other_files = []
                    break
    # -------------------------------------------------------

    # ---------- OTHER FILE TYPES (SEQUENTIAL) ----------
    for file in other_files:
        path = os.path.join(folder, file)
        name = file.lower()

        # -------- PT / PTH --------
        if (name.endswith(".pt") or name.endswith(".pth")) and TORCH_AVAILABLE and file != "cached_dataset.pt":
            print(f"[DATA] Loading PT: {file}")
            try:
                data = torch.load(path, map_location="cpu")
                if isinstance(data, list):
                    all_pos.extend(data)
                elif isinstance(data, dict) and "x" in data:
                    xs = torch.tensor(data["x"], dtype=torch.float32)
                    ys = torch.tensor(data["y"], dtype=torch.float32)
                    for x, y in zip(xs, ys):
                        all_pos.append((x, float(y)))
                else:
                    print(f"[DATA] Unrecognized PT format in {file}")
            except Exception as e:
                print(f"[DATA] Failed to load PT {file}: {e}")

        # -------- NPZ --------
        elif name.endswith(".npz") and TORCH_AVAILABLE:
            print(f"[DATA] Loading NPZ: {file}")
            try:
                npz = np.load(path)
                xs = torch.tensor(npz["x"], dtype=torch.float32)
                ys = torch.tensor(npz["y"], dtype=torch.float32)
                for x, y in zip(xs, ys):
                    all_pos.append((x, float(y)))
            except Exception as e:
                print(f"[DATA] Failed to load NPZ {file}: {e}")

        # -------- CSV --------
        elif name.endswith(".csv"):
            print(f"[DATA] Loading CSV: {file}")

            with open(path, encoding="utf-8") as f:
                header_line = f.readline().lower()

            if "moves" in header_line and "winner" in header_line and "id" in header_line:
                positions = load_lichess_games_csv(path, max_positions)
                all_pos.extend(positions)
            else:
                with open(path, encoding="utf-8") as f2:
                    reader = csv.reader(f2)
                    for row in reader:
                        if not row:
                            continue

                        # CASE 1: FEN,VALUE
                        if "/" in row[0] and len(row) >= 2:
                            fen = row[0]
                            val_str = row[-1]
                            try:
                                val = float(val_str)
                                b = chess.Board(fen)
                                x = board_to_tensor(b)
                                if x is not None:
                                    all_pos.append((x, val))
                            except Exception:
                                continue
                            continue

                        # CASE 2: numeric feature1..N + value
                        *feat, val_str = row
                        try:
                            val = float(val_str)
                        except ValueError:
                            continue

                        feat_floats = []
                        ok = True
                        for v in feat:
                            try:
                                feat_floats.append(float(v))
                            except ValueError:
                                ok = False
                                break

                        if not ok or not feat_floats:
                            continue

                        x = torch.tensor(feat_floats, dtype=torch.float32)
                        all_pos.append((x, val))

        else:
            print(f"[DATA] Skipping {file}")

        if max_positions is not None and len(all_pos) >= max_positions:
            print(f"[DATA] Reached max_positions={max_positions}")
            break
    # ---------------------------------------------------

    if max_positions is not None and len(all_pos) > max_positions:
        all_pos = all_pos[:max_positions]

    print(f"[DATA] Loaded {len(all_pos)} positions total.")

    # ---------- WRITE CACHE (SAFE) ----------
    if TORCH_AVAILABLE and len(all_pos) > 0:
        cache_path = os.path.join(folder, "cached_dataset.pt")
        try:
            sample_x = all_pos[0][0]
            feature_dim = int(sample_x.numel())
            num_pos = len(all_pos)
            bytes_needed = num_pos * feature_dim * 4  # float32
            gb_needed = bytes_needed / (1024 ** 3)
            print(f"[DATA] Cache estimate: ~{gb_needed:.2f} GB for {num_pos} positions.")

            MAX_CACHE_GB = 2.0
            if gb_needed > MAX_CACHE_GB:
                print(f"[DATA] Skipping cache save: estimated size {gb_needed:.2f} GB > {MAX_CACHE_GB} GB.")
            else:
                xs = torch.stack([x for x, _ in all_pos]).cpu()
                ys = torch.tensor([float(y) for _, y in all_pos], dtype=torch.float32).cpu()
                torch.save({"x": xs, "y": ys}, cache_path)
                print(f"[DATA] Saved cached dataset to {cache_path} ({len(all_pos)} positions).")
        except Exception as e:
            print(f"[DATA] Failed to save cached dataset: {e}")
    # ------------------------------

    return all_pos


# ============================================================
#  TRAINING SYSTEM
# ============================================================

if TORCH_AVAILABLE and SimpleValueNet is not None:

    class ReplayBuffer:
        def __init__(self, cap=500000):
            self.cap = cap
            self.data = []

        def add(self, x, y):
            if len(self.data) >= self.cap:
                self.data.pop(0)
            self.data.append((x, y))

        def extend(self, items):
            for x, y in items:
                self.add(x, y)

        def sample(self, n):
            batch = random.sample(self.data, min(n, len(self.data)))
            xs, ys = zip(*batch)
            xs = torch.stack(xs).to(DEVICE)
            ys = torch.tensor(ys, dtype=torch.float32).to(DEVICE)
            return xs, ys

        def __len__(self):
            return len(self.data)

    REPLAY = ReplayBuffer()

else:
    REPLAY = None


def find_stockfish_path() -> Optional[str]:
    """
    Try to find Stockfish executable in common locations.
    """
    # Common Stockfish executable names
    if platform.system() == "Windows":
        names = ["stockfish.exe", "stockfish-windows.exe", "stockfish_15.exe", "stockfish_16.exe"]
    else:
        names = ["stockfish", "stockfish-ubuntu", "stockfish-linux", "stockfish_15", "stockfish_16"]
    
    # Common paths to check
    paths = [
        "",  # Current directory
        "./",
        "../",
        "./engines/",
        "C:/stockfish/",
        "C:/Program Files/stockfish/",
        "/usr/local/bin/",
        "/usr/bin/",
        "/usr/games/",
        os.path.expanduser("~/stockfish/"),
    ]
    
    for path in paths:
        for name in names:
            full_path = os.path.join(path, name)
            if os.path.isfile(full_path):
                print(f"[ENGINE] Found Stockfish at: {full_path}")
                return full_path
    
    return None


def get_stockfish_engine() -> Optional[chess.engine.SimpleEngine]:
    """
    Initialize and return Stockfish engine.
    Returns None if Stockfish is not available.
    """
    global STOCKFISH_PATH
    
    if STOCKFISH_PATH is None:
        STOCKFISH_PATH = find_stockfish_path()
    
    if STOCKFISH_PATH is None:
        print("[ENGINE] Stockfish not found. Please install Stockfish or set STOCKFISH_PATH.")
        print("[ENGINE] Download from: https://stockfishchess.org/download/")
        return None
    
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        
        # Configure engine
        options = {}
        if STOCKFISH_SKILL_LEVEL is not None and STOCKFISH_ELO is None:
            options["Skill Level"] = STOCKFISH_SKILL_LEVEL
        
        if options:
            engine.configure(options)
            
        print(f"[ENGINE] Stockfish initialized with skill level: {STOCKFISH_SKILL_LEVEL}")
        return engine
    except Exception as e:
        print(f"[ENGINE] Failed to initialize Stockfish: {e}")
        return None


def choose_move_for_selfplay(board: Board) -> Optional[Move]:
    """
    Self-play move selection.
    Uses shallow search so training games are faster.
    """
    legal = list(board.generate_legal_moves())
    if not legal:
        return None

    move, _ = choose_move_with_search(board, legal_moves=legal, depth=SELFPLAY_DEPTH)
    return move


def play_game_vs_stockfish(
    engine: chess.engine.SimpleEngine,
    our_color: chess.Color = chess.WHITE,
    max_moves: int = 512
) -> Tuple[List[Tuple[torch.Tensor, float]], str, int]:
    """
    Play a game against Stockfish.
    Returns a tuple: (list of (features, target_value) from our perspective, result_str, move_count).
    Also collects simple per-game stats for printing/updating Elo outside the function.
    """
    if MODEL is None or not TORCH_AVAILABLE:
        return [], "0-1", 0

    board = Board()
    our_positions = []  # Positions where it was our turn
    moves_played = []
    start_time = time.time()

    for move_count in range(max_moves):
        if board.is_game_over():
            break

        # Store position if it's our turn
        if board.turn == our_color:
            x = board_to_tensor(board)
            if x is not None:
                our_positions.append(x)

        # Choose move
        if board.turn == our_color:
            # Our move (use training search depth)
            legal = list(board.generate_legal_moves())
            if not legal:
                break
            move, _ = choose_move_with_search(board, legal, depth=TRAIN_SEARCH_DEPTH)
            if move is None:
                break
        else:
            # Stockfish move
            try:
                result = engine.play(board, chess.engine.Limit(time=STOCKFISH_TIME_LIMIT))
                move = result.move
                if move is None:
                    break
            except Exception as e:
                print(f"[ENGINE] Stockfish error: {e}")
                break

        moves_played.append(move.uci())
        board.push(move)

    # Get game result
    res = board.result(claim_draw=True)
    duration = time.time() - start_time
    if res == "1-0":
        our_value = 1.0 if our_color == chess.WHITE else -1.0
    elif res == "0-1":
        our_value = -1.0 if our_color == chess.WHITE else 1.0
    else:
        our_value = 0.0

    # Label all our positions with the game outcome
    labeled = [(x, our_value) for x in our_positions]
    return labeled, res, len(moves_played)


def stockfish_training_games(num_games: int = 20) -> List[Tuple[torch.Tensor, float]]:
    """
    Play multiple games against Stockfish, alternating colors.
    Returns aggregated training data.
    """
    engine = get_stockfish_engine()
    if engine is None:
        print("[TRAIN] Stockfish not available; cannot run Stockfish training.")
        return []
    
    all_data = []
    wins = 0
    draws = 0
    losses = 0
    # We will adjust STOCKFISH_ELO dynamically after the batch; declare global early
    global STOCKFISH_ELO
    
    try:
        for i in range(num_games):
            # Alternate colors
            our_color = chess.WHITE if i % 2 == 0 else chess.BLACK
            color_name = "White" if our_color == chess.WHITE else "Black"
            print(f"[TRAIN] Game {i+1}/{num_games} vs Stockfish (playing as {color_name})...")

            game_data, res, move_count = play_game_vs_stockfish(engine, our_color)
            all_data.extend(game_data)

            # Update simple Elo estimate for the model
            global MODEL_ELO
            # Determine model score from result (1=win, 0.5=draw, 0=loss)
            if res == "1-0":
                model_score = 1.0 if our_color == chess.WHITE else 0.0
            elif res == "0-1":
                model_score = 1.0 if our_color == chess.BLACK else 0.0
            else:
                model_score = 0.5

            opponent_elo = STOCKFISH_ELO if STOCKFISH_ELO is not None else 2500.0
            expected = 1.0 / (1.0 + 10 ** ((opponent_elo - MODEL_ELO) / 400.0))
            old_elo = MODEL_ELO
            MODEL_ELO = MODEL_ELO + ELO_K * (model_score - expected)

            # Tally
            if model_score == 1.0:
                wins += 1
            elif model_score == 0.5:
                draws += 1
            else:
                losses += 1

            print(f"[TRAIN][GAME {i+1}] result={res}, color={color_name}, moves={move_count}, model_score={model_score}, expected={expected:.3f}")
            print(f"[TRAIN][GAME {i+1}] MODEL_ELO: {old_elo:.1f} -> {MODEL_ELO:.1f}  (Stockfish ELO={opponent_elo})")
            print(f"[TRAIN] Collected {len(game_data)} positions from game {i+1}")
            
    finally:
        engine.quit()
    
    print(f"[TRAIN] Batch summary: wins={wins}, draws={draws}, losses={losses} (out of {num_games})")
    
    # Dynamic Stockfish Elo adjustment
    # If model wins >60% of games, increase Stockfish Elo to make it harder
    total_games = wins + draws + losses
    if total_games > 0:
        win_rate = wins / total_games
        print(f"[TRAIN] Model win rate this batch: {win_rate:.1%}")
        
        old_sf_elo = STOCKFISH_ELO
        if win_rate > 0.6 and STOCKFISH_ELO < STOCKFISH_ELO_MAX:
            # Increase difficulty by ~50 Elo points per batch of strong wins
            elo_increase = min(50.0, STOCKFISH_ELO_MAX - STOCKFISH_ELO)
            STOCKFISH_ELO += elo_increase
            print(f"[TRAIN] Model performing well! Increasing Stockfish ELO: {old_sf_elo:.1f} -> {STOCKFISH_ELO:.1f}")
        elif win_rate < 0.3 and STOCKFISH_ELO > 1400.0:
            # Decrease difficulty if model is losing >70% of games
            elo_decrease = min(30.0, STOCKFISH_ELO - 1400.0)
            STOCKFISH_ELO -= elo_decrease
            print(f"[TRAIN] Model struggling. Decreasing Stockfish ELO: {old_sf_elo:.1f} -> {STOCKFISH_ELO:.1f}")
        else:
            print(f"[TRAIN] Stockfish ELO remains at {STOCKFISH_ELO:.1f} (win rate {win_rate:.1%} is in comfort zone)")
    
    return all_data


def self_play_game(max_moves=512):
    """
    Self-play one game using current MODEL, return list of (features, target_value).
    [DEPRECATED - Use stockfish_training_games instead]
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

        m = choose_move_for_selfplay(board)
        if m is None:
            break
        board.push(m)

    res = board.result(claim_draw=True)
    if res == "1-0":
        w, b = 1.0, -1.0
    elif res == "0-1":
        w, b = -1.0, 1.0
    else:
        w = b = 0.0

    labeled = []
    for turn, x in traj:
        labeled.append((x, w if turn else b))

    return labeled


def train_step(optimizer, batch=128):
    if MODEL is None or REPLAY is None or len(REPLAY) == 0:
        return None

    MODEL.train()
    xs, ys = REPLAY.sample(batch)
    preds = MODEL(xs)
    loss = F.mse_loss(preds, ys)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    MODEL.eval()
    return float(loss.item())


def evaluate_on_validation(val_x, val_y):
    if MODEL is None or val_x is None or val_y is None:
        return None
    MODEL.eval()
    with torch.no_grad():
        preds = MODEL(val_x)
        loss = F.mse_loss(preds, val_y)
    return float(loss.item())


def train_with_stockfish(
    dataset_folder="datasets",
    max_positions=100000,
    stockfish_games=20,
    steps_per_batch=5,
    batch=512,
    lr=1e-3,
):
    """
    Main training driver:
      1) load all datasets (cached if possible)
      2) split into train/val
      3) play games against Stockfish and train with replay
      4) save latest + best model (by val loss)

    If interrupted with Ctrl+C, the best model so far (if any) is
    saved to MODEL_PATH before exiting.
    """
    if not TORCH_AVAILABLE or SimpleValueNet is None:
        print("[TRAIN] Torch not available or model class missing.")
        return

    global MODEL
    if MODEL is None:
        init_model()
        if MODEL is None:
            print("[TRAIN] No model instance available.")
            return

    optimizer = torch.optim.Adam(MODEL.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # 1) load all datasets
    data = load_all_datasets(dataset_folder, max_positions)
    if not data:
        print("[TRAIN] No dataset positions loaded.")
        return

    # 2) split into train / validation
    random.shuffle(data)
    split_idx = int(0.9 * len(data)) if len(data) >= 10 else len(data)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    if REPLAY is not None:
        REPLAY.extend(train_data)
        print(f"[TRAIN] Replay size after dataset load: {len(REPLAY)}")
    else:
        print("[TRAIN] WARNING: No replay buffer, skipping training.")
        return

    if val_data and TORCH_AVAILABLE:
        val_x = torch.stack([x for x, _ in val_data]).to(DEVICE)
        val_y = torch.tensor([float(y) for _, y in val_data], dtype=torch.float32).to(DEVICE)
        print(f"[TRAIN] Validation set size: {len(val_data)}")
    else:
        val_x = None
        val_y = None
        print("[TRAIN] No validation set (too few positions).")

    best_val_loss = None
    best_state_dict = None

    try:
        # 3) Play games vs Stockfish in batches
        batch_size = 5  # Play 5 games at a time
        num_batches = (stockfish_games + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, stockfish_games)
            games_in_batch = end_idx - start_idx
            
            print(f"\n[TRAIN] Playing games {start_idx+1}-{end_idx} vs Stockfish...")
            
            # Play games against Stockfish
            stockfish_data = stockfish_training_games(games_in_batch)
            if REPLAY is not None and stockfish_data:
                REPLAY.extend(stockfish_data)
                print(f"[TRAIN] Added {len(stockfish_data)} positions from Stockfish games")
            
            # Train on accumulated data
            if REPLAY is not None and len(REPLAY) > 0:
                last_loss = None
                for _ in range(steps_per_batch * games_in_batch):
                    last_loss = train_step(optimizer, batch)
                
                val_loss = evaluate_on_validation(val_x, val_y) if val_x is not None else None
                if val_loss is not None:
                    print(f"[TRAIN] After batch {batch_idx+1}: train={last_loss:.6f}, val={val_loss:.6f}")
                    scheduler.step(val_loss)
                    
                    if best_val_loss is None or val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_state_dict = MODEL.state_dict()
                        torch.save(best_state_dict, BEST_MODEL_PATH)
                        print(f"[TRAIN] New best model saved to {BEST_MODEL_PATH} (val_loss={val_loss:.6f})")
                else:
                    print(f"[TRAIN] After batch {batch_idx+1}: train={last_loss:.6f}")
    
    except KeyboardInterrupt:
        print("\n[TRAIN] Training interrupted by user (Ctrl+C). Saving best model so far...")

    finally:
        if best_state_dict is not None:
            torch.save(best_state_dict, MODEL_PATH)
            print(f"[TRAIN] Saved best-so-far model to {MODEL_PATH}.")
        else:
            torch.save(MODEL.state_dict(), MODEL_PATH)
            print("[TRAIN] Saved latest model.pt (no best_val_loss recorded).")


# Keep the old function for backward compatibility
def train_with_datasets_and_selfplay(
    dataset_folder="datasets",
    max_positions=100000,
    selfplay_games=20,
    steps_per_game=5,
    batch=512,
    lr=1e-3,
):
    """
    [DEPRECATED] Use train_with_stockfish instead.
    Redirects to Stockfish training.
    """
    print("[TRAIN] Note: Self-play has been replaced with Stockfish training.")
    train_with_stockfish(
        dataset_folder=dataset_folder,
        max_positions=max_positions,
        stockfish_games=selfplay_games,
        steps_per_batch=steps_per_game,
        batch=batch,
        lr=lr
    )


# ============================================================
#  FUNCTION FOR NON-MODAL LOCAL EXECUTION
# ============================================================

def run_training_local(
    dataset_folder="datasets",
    max_positions=200000,
    stockfish_games=20,
    steps_per_batch=100,
    batch=512,
    lr=1e-3,
):
    """Wrapper for local training execution (used by Modal and direct execution)."""
    if not TORCH_AVAILABLE or SimpleValueNet is None:
        print("[TRAIN] Torch not installed or model missing — cannot train.")
        return
    train_with_stockfish(
        dataset_folder=dataset_folder,
        max_positions=max_positions,
        stockfish_games=stockfish_games,
        steps_per_batch=steps_per_batch,
        batch=batch,
        lr=lr,
    )


# ============================================================
#  RUN TRAINING IF EXECUTED DIRECTLY (NON-MODAL)
# ============================================================

if __name__ == "__main__":
    run_training_local(
        dataset_folder="datasets",
        max_positions=200000,
        stockfish_games=20,
        steps_per_batch=100,
        batch=512,
        lr=1e-3,
    )
