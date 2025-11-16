# ============================================================
#  FAST + STABLE CPU CHESS BOT (LIMITED SEARCH, NN EVAL)
#  - Single-threaded CPU only (no GPU)
#  - NN eval + tiny classical stabilizer
#  - Small alpha-beta search (3 ply)
#  - Hard GLOBAL NODE LIMIT to prevent hangs
#  - No training, no dataset loading, no Stockfish
# ============================================================

from .utils import chess_manager, GameContext
from chess import Board, Move
import chess
import random
import math
import os
from typing import Optional, List

# ============================================================
#  TORCH SETUP
# ============================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

# ============================================================
#  CONSTANTS + PATHS
# ============================================================

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

BOARD_SIZE = 8
NUM_PIECE_PLANES = 12
NUM_EXTRA_PLANES = 1
NUM_PLANES = NUM_PIECE_PLANES + NUM_EXTRA_PLANES

FEATURE_DIM = NUM_PLANES * BOARD_SIZE * BOARD_SIZE + 4

MODEL = None
DEVICE = "cpu"
MODEL_PATH = os.path.join(ROOT_DIR, "src", "weights", "model.pt")

# Search parameters
SEARCH_DEPTH = 2
NODE_LIMIT = 8000
nodes = 0

# ============================================================
#  ENCODING BOARD TO TENSOR
# ============================================================

def piece_plane_index(piece):
    return (0 if piece.color else 6) + (piece.piece_type - 1)

def board_to_tensor(board):
    if not TORCH_AVAILABLE:
        return None

    planes = torch.zeros(NUM_PLANES, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)

    for sq, piece in board.piece_map().items():
        r = sq // 8
        c = sq % 8
        planes[piece_plane_index(piece), r, c] = 1.0

    planes[NUM_PIECE_PLANES, :, :] = 1.0 if board.turn else 0.0

    flat = planes.reshape(-1)
    castling = torch.tensor([
        float(board.has_kingside_castling_rights(True)),
        float(board.has_queenside_castling_rights(True)),
        float(board.has_kingside_castling_rights(False)),
        float(board.has_queenside_castling_rights(False)),
    ], dtype=torch.float32)

    return torch.cat([flat, castling], dim=0)

# ============================================================
#  CNN VALUE NETWORK
# ============================================================

if TORCH_AVAILABLE:

    class CnnValueNet(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            planes = NUM_PLANES
            self.board_dim = planes * 64
            self.meta_dim = input_dim - self.board_dim

            self.conv1 = nn.Conv2d(planes, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 64, 3, padding=1)

            self.fc1 = nn.Linear(64 * 64 + self.meta_dim, 512)
            self.fc2 = nn.Linear(512, 128)
            self.fc_out = nn.Linear(128, 1)
            self.drop = nn.Dropout(0.2)

        def forward(self, x):
            b = x[:, :self.board_dim].view(-1, NUM_PLANES, 8, 8)
            m = x[:, self.board_dim:]
            h = F.relu(self.conv1(b))
            h = F.relu(self.conv2(h))
            h = F.relu(self.conv3(h))
            h = h.view(h.size(0), -1)
            h = torch.cat([h, m], dim=1)
            h = F.relu(self.fc1(h))
            h = self.drop(h)
            h = F.relu(self.fc2(h))
            return torch.tanh(self.fc_out(h)).squeeze(-1)

def init_model():
    global MODEL

    if not TORCH_AVAILABLE:
        MODEL = None
        print("[ML] Torch unavailable, using classical eval only.")
        return

    MODEL = CnnValueNet(FEATURE_DIM).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        try:
            state = torch.load(MODEL_PATH, map_location=DEVICE)
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            MODEL.load_state_dict(state)
            print("[ML] Model loaded")
        except Exception as e:
            print("[ML] Failed to load model:", e)
    else:
        print("[ML] No model found, random weights")

    MODEL.eval()

init_model()

# ============================================================
#  CLASSICAL EVAL
# ============================================================

VALUES = {
    chess.PAWN:100,
    chess.KNIGHT:320,
    chess.BISHOP:330,
    chess.ROOK:500,
    chess.QUEEN:900,
    chess.KING:0,
}

def classical_eval(board):
    s = sum((VALUES[p.piece_type] * (1 if p.color else -1)) for p in board.piece_map().values())
    s = max(-2000, min(2000, s)) / 2000.0
    return s if board.turn else -s

# ============================================================
#  FINAL EVAL (NN + classical)
# ============================================================

def evaluate(board):
    c = classical_eval(board)

    if MODEL is None:
        return c

    x = board_to_tensor(board)
    if x is None:
        return c

    with torch.no_grad():
        v = MODEL(x.to(DEVICE).unsqueeze(0))[0].item()

    return 0.2*c + 0.8*v

# ============================================================
#  MOVE ORDERING
# ============================================================

def order_moves(board):
    moves = list(board.generate_legal_moves())
    return sorted(
        moves,
        key=lambda m: (
            board.is_capture(m),
            m.promotion is not None,
            board.gives_check(m)
        ),
        reverse=True
    )

# ============================================================
#  LIMITED NEGAMAX
# ============================================================

def negamax(board, depth, alpha, beta):
    global nodes

    if nodes >= NODE_LIMIT:
        return evaluate(board)

    if depth == 0 or board.is_game_over():
        return evaluate(board)

    nodes += 1
    best = -float("inf")

    for move in order_moves(board):
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()

        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break

    return best

# ============================================================
#  SEARCH WRAPPER
# ============================================================

def choose_move(board, legal_moves):
    if not legal_moves:
        return None, {}

    if MODEL is None:
        return random.choice(legal_moves), {}

    scores = []
    shuffled = list(legal_moves)
    random.shuffle(shuffled)

    global nodes
    for move in shuffled:
        board.push(move)
        nodes = 0
        s = -negamax(board, SEARCH_DEPTH - 1, -float("inf"), float("inf"))
        board.pop()
        scores.append(s)

    best_idx = max(range(len(scores)), key=scores.__getitem__)
    best = shuffled[best_idx]

    probs = softmax(scores)
    return best, {m:p for m,p in zip(shuffled, probs)}

def softmax(scores):
    m = max(scores)
    exps = [math.exp(s-m) for s in scores]
    tot = sum(exps)
    return [x/tot for x in exps]

# ============================================================
#  CHESSHACKS ENTRYPOINTS
# ============================================================

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    moves = list(ctx.board.generate_legal_moves())
    if not moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves")
    m, probs = choose_move(ctx.board, moves)
    ctx.logProbabilities(probs)
    return m

@chess_manager.reset
def reset_func(ctx: GameContext):
    pass
