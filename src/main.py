from .utils import chess_manager, GameContext
from chess import Board, Move
import chess
import random
import time
import math
import os

# ============================================================
#  OPTIONAL ML IMPORTS (Torch). CODE STILL RUNS WITHOUT IT.
# ============================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception as e:
    print("[WARN] Torch not available, running without ML:", e)
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

# ============================================================
#  FEATURE ENCODING
# ============================================================

BOARD_SIZE = 8
NUM_PIECE_PLANES = 12      # 6 piece types * 2 colors
NUM_EXTRA_PLANES = 1       # side to move plane
FEATURE_DIM = BOARD_SIZE * BOARD_SIZE * (NUM_PIECE_PLANES + NUM_EXTRA_PLANES) + 4
# 4 = castling rights flags


def piece_plane_index(piece: chess.Piece) -> int:
    """
    Map a python-chess Piece to a plane index in [0, 11].

    Planes:
      0..5   = white pawn, knight, bishop, rook, queen, king
      6..11  = black pawn, knight, bishop, rook, queen, king
    """
    base = 0 if piece.color else 6
    return base + (piece.piece_type - 1)


def board_to_tensor(board: chess.Board):
    """
    Convert a board to a 1D float tensor of size FEATURE_DIM.
    Returns None if torch is not available.
    """
    if not TORCH_AVAILABLE or torch is None:
        return None

    planes = torch.zeros(
        NUM_PIECE_PLANES + NUM_EXTRA_PLANES,
        BOARD_SIZE,
        BOARD_SIZE,
        dtype=torch.float32,
    )

    # Piece planes
    for square, piece in board.piece_map().items():
        rank = square // 8  # 0..7
        file = square % 8   # 0..7
        plane_idx = piece_plane_index(piece)
        planes[plane_idx, rank, file] = 1.0

    # Side-to-move plane (last plane)
    side_plane_idx = NUM_PIECE_PLANES
    planes[side_plane_idx, :, :] = 1.0 if board.turn else 0.0

    flat = planes.reshape(-1)

    # Castling rights flags
    castle_flags = torch.tensor([
        float(board.has_kingside_castling_rights(True)),   # White O-O
        float(board.has_queenside_castling_rights(True)),  # White O-O-O
        float(board.has_kingside_castling_rights(False)),  # Black O-O
        float(board.has_queenside_castling_rights(False)), # Black O-O-O
    ], dtype=torch.float32)

    return torch.cat([flat, castle_flags], dim=0)  # shape [FEATURE_DIM]


# ============================================================
#  MODEL
# ============================================================

if TORCH_AVAILABLE and nn is not None:

    class SimpleValueNet(nn.Module):
        """
        Tiny value network: input = board features, output = scalar in [-1, 1].
        Positive = good for side to move, negative = bad.
        """
        def __init__(self, input_dim: int):
            super().__init__()
            hidden = 256
            self.fc1 = nn.Linear(input_dim, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.out = nn.Linear(hidden, 1)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return torch.tanh(self.out(x)).squeeze(-1)  # [batch]

else:
    SimpleValueNet = None  # just to keep the name defined


MODEL = None
DEVICE = "cpu"
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
MODEL_PATH = os.path.join(ROOT_DIR, "model.pt")


def init_model():
    """
    Initialize MODEL from disk if Torch is available.
    """
    global MODEL, DEVICE

    if not TORCH_AVAILABLE or SimpleValueNet is None:
        print("[ML] Torch not available, using non-ML move selection.")
        MODEL = None
        return

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL = SimpleValueNet(FEATURE_DIM).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        try:
            state = torch.load(MODEL_PATH, map_location=DEVICE)
            MODEL.load_state_dict(state)
            print(f"[ML] Loaded model from {MODEL_PATH}")
        except Exception as e:
            print(f"[ML] Failed to load model.pt, using fresh weights: {e}")
    else:
        print("[ML] No model.pt found, starting from random weights.")

    MODEL.eval()


init_model()


# ============================================================
#  EVALUATION + MOVE SELECTION
# ============================================================

def softmax(scores):
    """
    Numerically stable softmax over a list of floats.
    Returns a list of probabilities that sum to 1.
    """
    if not scores:
        return []

    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(scores)] * len(scores)
    return [e / total for e in exps]


def evaluate_position(board: chess.Board) -> float:
    """
    Evaluate board from side-to-move’s POV using the neural net.
    If no ML, returns 0.0 (neutral).
    """
    if not TORCH_AVAILABLE or MODEL is None:
        return 0.0

    x = board_to_tensor(board)
    if x is None:
        return 0.0

    with torch.no_grad():
        x = x.to(DEVICE).unsqueeze(0)  # [1, FEATURE_DIM]
        v = MODEL(x)[0].item()
    return float(v)


def choose_move_ml(board: chess.Board, legal_moves):
    """
    Choose a move using ML evaluation.
    If ML not available, falls back to random.
    """
    if not TORCH_AVAILABLE or MODEL is None:
        return random.choice(legal_moves), {m: 1.0 / len(legal_moves) for m in legal_moves}

    scores = []
    for move in legal_moves:
        board.push(move)
        s = evaluate_position(board)
        board.pop()
        scores.append(s)

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
    Called by ChessHacks every time the bot needs to move.
    Must return a legal python-chess Move.
    """

    print("Cooking move...")
    print(ctx.board.move_stack)
    time.sleep(0.05)

    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (game is over).")

    move, prob_map = choose_move_ml(ctx.board, legal_moves)
    ctx.logProbabilities(prob_map)
    return move


@chess_manager.reset
def reset_func(ctx: GameContext):
    """
    Called when a new game starts. Reset per-game state here if needed.
    """
    # nothing persistent per game yet
    pass


# ============================================================
#  DATASET LOADING (PGN)
# ============================================================

def load_pgn_dataset(path, max_positions=None):
    """
    Load positions from a PGN file.
    Each position is turned into (features, value_target) where:
        1-0  -> +1
        0-1  -> -1
        other/draw -> 0
    """
    if not TORCH_AVAILABLE:
        print("[DATA] Torch not available; cannot load PGN dataset into tensors.")
        return []

    import chess.pgn

    positions = []
    count_games = 0

    with open(path, "r", encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            count_games += 1

            result_str = game.headers.get("Result", "*")
            if result_str == "1-0":
                outcome = 1.0
            elif result_str == "0-1":
                outcome = -1.0
            else:
                outcome = 0.0

            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                x = board_to_tensor(board)
                if x is None:
                    continue
                positions.append((x, outcome))

                if max_positions is not None and len(positions) >= max_positions:
                    print(f"[DATA] Reached max_positions={max_positions} after {count_games} games.")
                    return positions

    print(f"[DATA] Loaded {len(positions)} positions from {count_games} games in {path}")
    return positions


# ============================================================
#  TRAINING STUFF (ONLY ACTIVE IF TORCH AVAILABLE)
# ============================================================

if TORCH_AVAILABLE and SimpleValueNet is not None:

    class ReplayBuffer:
        def __init__(self, capacity=50000):
            self.capacity = capacity
            self.data = []

        def add(self, x, y):
            # x: tensor [FEATURE_DIM], y: float
            if len(self.data) >= self.capacity:
                self.data.pop(0)
            self.data.append((x, y))

        def extend(self, pairs):
            for x, y in pairs:
                self.add(x, y)

        def sample_batch(self, batch_size):
            batch = random.sample(self.data, min(batch_size, len(self.data)))
            xs, ys = zip(*batch)
            xs = torch.stack(xs, dim=0)
            ys = torch.tensor(ys, dtype=torch.float32)
            return xs.to(DEVICE), ys.to(DEVICE)

        def __len__(self):
            return len(self.data)


    REPLAY_BUFFER = ReplayBuffer(capacity=100000)


    def choose_move_for_selfplay(board: chess.Board) -> Move | None:
        legal_moves = list(board.generate_legal_moves())
        if not legal_moves:
            return None
        move, _ = choose_move_ml(board, legal_moves)
        return move


    def self_play_one_game(max_moves=512):
        """
        Self-play game using current MODEL.
        Returns list of (features_tensor, value_target).
        """
        if MODEL is None:
            print("[TRAIN] MODEL is None; cannot self-play.")
            return []

        board = Board()
        trajectory = []

        # Play until game over or too long
        for _ in range(max_moves):
            if board.is_game_over():
                break

            x = board_to_tensor(board)
            if x is None:
                break

            trajectory.append((board.turn, x))

            move = choose_move_for_selfplay(board)
            if move is None:
                break
            board.push(move)

        # Final result
        if board.is_game_over():
            result_str = board.result(claim_draw=True)
        else:
            result_str = "*"

        if result_str == "1-0":
            white_outcome, black_outcome = 1.0, -1.0
        elif result_str == "0-1":
            white_outcome, black_outcome = -1.0, 1.0
        else:
            white_outcome = black_outcome = 0.0

        labeled_positions = []
        for side_to_move, x in trajectory:
            if side_to_move:  # white to move
                labeled_positions.append((x, white_outcome))
            else:
                labeled_positions.append((x, black_outcome))

        return labeled_positions


    def train_step(optimizer, batch_size=128):
        if MODEL is None or len(REPLAY_BUFFER) == 0:
            return None

        MODEL.train()
        xs, ys = REPLAY_BUFFER.sample_batch(batch_size)
        preds = MODEL(xs)
        loss = F.mse_loss(preds, ys)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        MODEL.eval()
        return loss.item()


    def train_with_dataset_and_selfplay(
        pgn_path: str | None = None,
        max_dataset_positions: int | None = None,
        selfplay_games: int = 50,
        train_steps_per_game: int = 10,
        batch_size: int = 128,
        lr: float = 1e-3,
    ):
        """
        High-level training loop:
          1. Optionally load PGN dataset into replay buffer.
          2. Run self-play games to generate more data.
          3. Train the value net.
          4. Save updated model.pt
        """
        global MODEL

        if MODEL is None:
            print("[TRAIN] MODEL is None, re-initializing.")
            init_model()
            if MODEL is None:
                print("[TRAIN] Still no MODEL; aborting.")
                return

        optimizer = torch.optim.Adam(MODEL.parameters(), lr=lr)

        # 1. Dataset
        if pgn_path is not None:
            full_path = pgn_path
            if not os.path.isabs(full_path):
                full_path = os.path.join(ROOT_DIR, pgn_path)
            if os.path.exists(full_path):
                dataset_positions = load_pgn_dataset(full_path, max_positions=max_dataset_positions)
                REPLAY_BUFFER.extend(dataset_positions)
                print(f"[TRAIN] Replay buffer size after PGN load: {len(REPLAY_BUFFER)}")
            else:
                print(f"[TRAIN] PGN path not found: {full_path}")

        # 2–3. Self-play + training
        total_positions = len(REPLAY_BUFFER)
        last_loss = None

        for g in range(1, selfplay_games + 1):
            positions = self_play_one_game()
            REPLAY_BUFFER.extend(positions)
            total_positions += len(positions)
            print(f"[TRAIN] Game {g}/{selfplay_games} -> {len(positions)} positions (replay size={len(REPLAY_BUFFER)})")

            for _ in range(train_steps_per_game):
                loss = train_step(optimizer, batch_size=batch_size)
                if loss is not None:
                    last_loss = loss

            if last_loss is not None:
                print(f"[TRAIN] Last loss after game {g}: {last_loss:.4f}")

        # 4. Save
        torch.save(MODEL.state_dict(), MODEL_PATH)
        print(f"[TRAIN] Saved updated model to {MODEL_PATH}")
        print(f"[TRAIN] Total positions seen this run: {total_positions}")


# ============================================================
#  CLI ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    if not TORCH_AVAILABLE or SimpleValueNet is None:
        print("Torch not available; training is disabled. Bot will still run with random-ish moves.")
    else:
        # Example training run
        # Put your PGN at e.g. <project-root>/datasets/data.pgn
        example_pgn = os.path.join("datasets", "data.pgn")  # change or set to None

        train_with_dataset_and_selfplay(
            pgn_path=example_pgn,        # or None if no PGN yet
            max_dataset_positions=20000,
            selfplay_games=50,
            train_steps_per_game=10,
            batch_size=128,
            lr=1e-3,
        )
