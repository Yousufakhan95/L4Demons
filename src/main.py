# ============================================================
#  FULL CHESSHACKS BOT + TRAINING + UNIVERSAL DATA LOADER
#  (GPU, negamax+alpha-beta, parallel PGN loading, replay buf,
#   upgraded net, validation, LR scheduler, best-model saving)
# ============================================================

from .utils import chess_manager, GameContext
from chess import Board, Move
import chess
import random
import math
import os
import csv
import numpy as np
from typing import Optional
import concurrent.futures

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
SEARCH_DEPTH = 3          # main engine depth (root)
SELFPLAY_DEPTH = 1        # depth used during self-play (keep small for speed)

# Data config
MIN_FULLMOVES_FOR_PGN_LABEL = 10  # only label positions after move 10 to reduce early-game noise


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
#  CLASSICAL CHESS EVALUATION (MATERIAL + SIMPLE HEURISTICS)
# ============================================================

# Piece values in centipawns
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   0,    # king's "value" is handled via safety, not material
}

# Central squares for basic center control
CENTER_SQUARES_STRONG = [
    chess.D4, chess.E4, chess.D5, chess.E5
]
CENTER_SQUARES_EXTENDED = [
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.F4,
    chess.C5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
]


def material_score(board: chess.Board) -> int:
    """
    Material in centipawns from White's POV.
    White pieces positive, Black pieces negative.
    """
    score = 0
    for square, piece in board.piece_map().items():
        val = PIECE_VALUES.get(piece.piece_type, 0)
        if piece.color == chess.WHITE:
            score += val
        else:
            score -= val
    return score


def piece_activity_score(board: chess.Board) -> int:
    """
    Very simple activity / centralization heuristic for minor/major pieces.
    Encourages knights/bishops/queens toward the center.
    """
    score = 0
    for square, piece in board.piece_map().items():
        if piece.piece_type not in (chess.KNIGHT, chess.BISHOP, chess.QUEEN):
            continue

        file = chess.square_file(square)  # 0..7
        rank = chess.square_rank(square)  # 0..7

        # Distance from edge: more central = bigger bonus
        dist_file = min(file, 7 - file)
        dist_rank = min(rank, 7 - rank)
        activity = dist_file + dist_rank  # max 6

        bonus = activity * 2  # up to ~12 cp
        if piece.color == chess.WHITE:
            score += bonus
        else:
            score -= bonus

    return score


def center_control_score(board: chess.Board) -> int:
    """
    Reward occupying strong/extended center squares (very rough).
    """
    score = 0
    for sq in CENTER_SQUARES_STRONG:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        bonus = 15  # strong center
        if piece.piece_type == chess.PAWN:
            bonus += 10  # central pawns are huge

        if piece.color == chess.WHITE:
            score += bonus
        else:
            score -= bonus

    for sq in CENTER_SQUARES_EXTENDED:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        bonus = 8
        if piece.piece_type == chess.PAWN:
            bonus += 6

        if piece.color == chess.WHITE:
            score += bonus
        else:
            score -= bonus

    return score


def development_score(board: chess.Board) -> int:
    """
    Very crude development heuristic:
      - Penalize knights/bishops still on their original squares
      - Reward castling
      - Penalize staying uncastled in the center after move 10
    """
    score = 0
    fullmove = board.fullmove_number

    # --- White undeveloped minors ---
    piece_b1 = board.piece_at(chess.B1)
    if piece_b1 is not None and piece_b1.color == chess.WHITE and piece_b1.piece_type == chess.KNIGHT:
        score -= 12

    piece_g1 = board.piece_at(chess.G1)
    if piece_g1 is not None and piece_g1.color == chess.WHITE and piece_g1.piece_type == chess.KNIGHT:
        score -= 12

    piece_c1 = board.piece_at(chess.C1)
    if piece_c1 is not None and piece_c1.color == chess.WHITE and piece_c1.piece_type == chess.BISHOP:
        score -= 10

    piece_f1 = board.piece_at(chess.F1)
    if piece_f1 is not None and piece_f1.color == chess.WHITE and piece_f1.piece_type == chess.BISHOP:
        score -= 10

    # --- Black undeveloped minors ---
    piece_b8 = board.piece_at(chess.B8)
    if piece_b8 is not None and piece_b8.color == chess.BLACK and piece_b8.piece_type == chess.KNIGHT:
        score += 12  # from White POV, black undeveloped is good for White

    piece_g8 = board.piece_at(chess.G8)
    if piece_g8 is not None and piece_g8.color == chess.BLACK and piece_g8.piece_type == chess.KNIGHT:
        score += 12

    piece_c8 = board.piece_at(chess.C8)
    if piece_c8 is not None and piece_c8.color == chess.BLACK and piece_c8.piece_type == chess.BISHOP:
        score += 10

    piece_f8 = board.piece_at(chess.F8)
    if piece_f8 is not None and piece_f8.color == chess.BLACK and piece_f8.piece_type == chess.BISHOP:
        score += 10

    # --- Castling / king in center ---
    wk_sq = board.king(chess.WHITE)
    bk_sq = board.king(chess.BLACK)

    # White king castled
    if wk_sq in (chess.G1, chess.C1):
        score += 20
    # Black king castled
    if bk_sq in (chess.G8, chess.C8):
        score -= 20

    # If we’re past move 10 and kings are still in the center, penalize that
    if fullmove >= 10:
        if wk_sq in (chess.E1, chess.D1):
            score -= 20
        if bk_sq in (chess.E8, chess.D8):
            score += 20

    return score


def king_safety_score(board: chess.Board) -> int:
    """
    Tiny king safety heuristic:
      - Reward having pawn shield in front of castled king
      - Penalize wide-open king files next to it
    """
    score = 0
    wk_sq = board.king(chess.WHITE)
    bk_sq = board.king(chess.BLACK)

    # White king typical castled square
    if wk_sq == chess.G1:
        # pawns f2, g2, h2 as shield
        for sq in (chess.F2, chess.G2, chess.H2):
            p = board.piece_at(sq)
            if p is not None and p.color == chess.WHITE and p.piece_type == chess.PAWN:
                score += 4
            else:
                score -= 3  # weak spot

    # Black king typical castled square
    if bk_sq == chess.G8:
        for sq in (chess.F7, chess.G7, chess.H7):
            p = board.piece_at(sq)
            if p is not None and p.color == chess.BLACK and p.piece_type == chess.PAWN:
                score -= 4  # good for Black = bad for White
            else:
                score += 3

    return score


def early_pawn_weirdness_score(board: chess.Board) -> int:
    """
    Penalize dumb early pawn moves:
      - Big penalty for pushing the f-pawn too far while king is in the center.
      - Smaller penalty for randomly advancing flank pawns in the opening.
    Returns score in centipawns from White's POV.
    """
    fullmove = board.fullmove_number
    if fullmove > 10:
        return 0  # only care in the opening

    score = 0

    # ----- f-pawn penalties -----
    F_PAWN_PENALTY_CP = 40   # 0.4 pawn
    FLANK_PAWN_PENALTY_CP = 15

    piece_map = board.piece_map()

    # White f-pawn
    white_f_pawn_square = None
    for sq, pc in piece_map.items():
        if pc.color and pc.piece_type == chess.PAWN and chess.square_file(sq) == chess.FILE_F:
            white_f_pawn_square = sq
            break

    if white_f_pawn_square is not None:
        rank = chess.square_rank(white_f_pawn_square)  # 0=rank1, ..., 7=rank8
        # if pawn is beyond f3 (rank >= 3) while king still on e1 -> bad
        if rank >= 3 and board.king(chess.WHITE) == chess.E1:
            score -= F_PAWN_PENALTY_CP

    # Black f-pawn
    black_f_pawn_square = None
    for sq, pc in piece_map.items():
        if (not pc.color) and pc.piece_type == chess.PAWN and chess.square_file(sq) == chess.FILE_F:
            black_f_pawn_square = sq
            break

    if black_f_pawn_square is not None:
        rank = chess.square_rank(black_f_pawn_square)
        # from White POV, if black f-pawn is advanced (rank <= 4) with king on e8, that's good
        if rank <= 4 and board.king(chess.BLACK) == chess.E8:
            score += F_PAWN_PENALTY_CP

    # ----- flank pawns (a, b, g, h) randomly pushed -----
    for sq, pc in piece_map.items():
        if pc.piece_type != chess.PAWN:
            continue
        file_idx = chess.square_file(sq)
        rank_idx = chess.square_rank(sq)

        if file_idx in (chess.FILE_A, chess.FILE_B, chess.FILE_G, chess.FILE_H):
            # White flank pawns advanced beyond rank 2
            if pc.color and rank_idx > 1:
                score -= FLANK_PAWN_PENALTY_CP
            # Black flank pawns advanced beyond rank 7 (toward White side)
            if (not pc.color) and rank_idx < 6:
                score += FLANK_PAWN_PENALTY_CP

    return score


def classical_eval(board: chess.Board) -> float:
    """
    Classical evaluation from POV of side-to-move in [-1, 1].
    Combines:
      - material
      - piece activity / centralization
      - center control (occupancy)
      - development
      - king safety
      - early pawn weirdness (f-pawn / flanks)
    """
    # All these scores are "White minus Black" in centipawns
    score_cp = 0
    score_cp += material_score(board)
    score_cp += piece_activity_score(board)
    score_cp += center_control_score(board)
    score_cp += development_score(board)
    score_cp += king_safety_score(board)
    score_cp += early_pawn_weirdness_score(board)

    # Clamp and normalize to [-1, 1]
    MAX_CP = 2000.0  # ~20 pawns equivalent; very generous
    if score_cp > MAX_CP:
        score_cp = MAX_CP
    elif score_cp < -MAX_CP:
        score_cp = -MAX_CP

    value = score_cp / MAX_CP  # now in roughly [-1, 1]

    # Convert to side-to-move POV
    if board.turn == chess.WHITE:
        return float(value)
    else:
        return float(-value)


def evaluate_position(board: chess.Board) -> float:
    """
    Final evaluation from POV of side to move.

    Combines:
      - classical_eval(board): basic chess heuristics
      - NN value (if available): learned evaluation

    Both are in [-1, 1], then blended.
    """
    classical = classical_eval(board)

    if not TORCH_AVAILABLE or MODEL is None:
        return classical

    x = board_to_tensor(board)
    if x is None:
        return classical

    with torch.no_grad():
        x = x.to(DEVICE).unsqueeze(0)
        v = MODEL(x)[0].item()

    # Make hand-crafted chess knowledge dominate more
    ALPHA = 0.7  # classical weight
    BETA = 0.3   # NN weight

    combined = ALPHA * classical + BETA * float(v)
    return combined


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
    if depth is None:
        depth = SEARCH_DEPTH

    if not legal_moves:
        return None, {}

    # If NN not available, use classical_eval in a 1-ply search
    if not TORCH_AVAILABLE or MODEL is None:
        scores = []
        best_score = -float("inf")
        best_move = None

        for move in legal_moves:
            board.push(move)
            s = classical_eval(board)
            board.pop()
            scores.append(s)

            if s > best_score or best_move is None:
                best_score = s
                best_move = move

        probs = softmax(scores)
        prob_map = {m: p for m, p in zip(legal_moves, probs)}
        return best_move, prob_map

    # Normal NN+classical search
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
    Use eval (classical + NN) to pick a move via 1-ply evaluation.
    - If temperature is None or <= 0 -> pure greedy (argmax on eval).
    - If temperature > 0 -> sample from softmax(scores / temperature).
    """
    if not legal_moves:
        return None, {}

    scores = []
    for move in legal_moves:
        board.push(move)
        s = evaluate_position(board)
        board.pop()
        scores.append(s)

    # Greedy / deterministic mode
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
    Uses negamax + alpha-beta for stronger play (NN + classical eval).
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


def self_play_game(max_moves=512):
    """
    Self-play one game using current MODEL, return list of (features, target_value).
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


def train_with_datasets_and_selfplay(
    dataset_folder="datasets",
    max_positions=100000,
    selfplay_games=20,
    steps_per_game=5,
    batch=512,
    lr=1e-3,
):
    """
    Main training driver:
      1) load all datasets (cached if possible)
      2) split into train/val
      3) do self-play and train with replay
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
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
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
        # 3) self-play + train
        for g in range(1, selfplay_games + 1):
            pos = self_play_game()
            if REPLAY is not None:
                REPLAY.extend(pos)
            print(f"[TRAIN] Game {g}/{selfplay_games} — {len(pos)} self-play positions.")

            last_loss = None
            for _ in range(steps_per_game):
                last_loss = train_step(optimizer, batch)

            val_loss = evaluate_on_validation(val_x, val_y) if val_x is not None else None
            if val_loss is not None:
                print(f"[TRAIN] Loss after game {g}: train={last_loss:.6f}, val={val_loss:.6f}")
                scheduler.step(val_loss)

                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = MODEL.state_dict()
                    torch.save(best_state_dict, BEST_MODEL_PATH)
                    print(f"[TRAIN] New best model saved to {BEST_MODEL_PATH} (val_loss={val_loss:.6f})")
            else:
                print(f"[TRAIN] Loss after game {g}: train={last_loss:.6f}")

    except KeyboardInterrupt:
        print("\n[TRAIN] Training interrupted by user (Ctrl+C). Saving best model so far...")

    finally:
        if best_state_dict is not None:
            torch.save(best_state_dict, MODEL_PATH)
            print(f"[TRAIN] Saved best-so-far model to {MODEL_PATH}.")
        else:
            torch.save(MODEL.state_dict(), MODEL_PATH)
            print("[TRAIN] Saved latest model.pt (no best_val_loss recorded).")


# ============================================================
#  RUN TRAINING IF EXECUTED DIRECTLY
# ============================================================

if __name__ == "__main__":
    if not TORCH_AVAILABLE or SimpleValueNet is None:
        print("[TRAIN] Torch not installed or model missing — cannot train.")
    else:
        train_with_datasets_and_selfplay(
            dataset_folder="datasets",
            max_positions=200000,
            selfplay_games=20,
            steps_per_game=100,
            batch=512,
            lr=1e-3,
        )
