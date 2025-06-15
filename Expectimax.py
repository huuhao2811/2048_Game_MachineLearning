import pygame
import random
import math
import copy
import numpy as np
from collections import defaultdict

pygame.init()

FPS = 500
WIDTH, HEIGHT = 800, 800
ROWS = 4
COLS = 4

RECT_HEIGHT = HEIGHT // ROWS
RECT_WIDTH = WIDTH // COLS

OUTLINE_COLOR = (120, 120, 120)
OUTLINE_THICHKNESS = 10
BACKGROUND_COLOR = (0, 0, 0)
FONT_COLOR = (255, 255, 0)

FONT = pygame.font.SysFont("comicsans", 70, bold=True)
MOVE_VEL = 20

WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2048")

# Constants for Expectimax
PLAYER = 0
BOARD = 1
MAX_DEPTH = 4  # Độ sâu tìm kiếm tối đa

# Trọng số cho các heuristic
HEURISTIC_WEIGHTS = {
    'empty_tiles': 537.4542494459174,
    'max_in_corner': 959.2308181914565,
    'merge_potential': 810.6879021182804,  # Tăng trọng số để ưu tiên sát nhập
    'smoothness': 1261.902509904945,
    'monotonicity': 1640.0810210208008,
    'big_tile_group': 285.07012676558884,
    'total_tile_value': 1642.0989852437706,
    'high_value_merge': 909.5970981704978  # Thêm trọng số cho sát nhập khối lớn
}

class Tile:
    COLORS = [
        (220, 220, 220),
        (192, 192, 192),
        (128, 128, 128),
        (105, 105, 105),
        (79, 79, 79),
        (54, 54, 54),
        (38, 38, 38),
        (0, 0, 0)
    ]

    def __init__(self, value, row, col):
        self.value = value
        self.row = row
        self.col = col
        self.x = col * RECT_WIDTH
        self.y = row * RECT_HEIGHT

    def get_color(self):
        if self.value > 128:
            color_index = int(math.log2(self.value)) - 8
        else:
            color_index = int(math.log2(self.value)) - 1
        color = self.COLORS[color_index] if color_index < len(self.COLORS) else (0, 0, 0)
        return color

    def draw(self, window):
        color = self.get_color()
        pygame.draw.rect(window, color, (self.x, self.y, RECT_WIDTH, RECT_HEIGHT))
        text = FONT.render(str(self.value), True, FONT_COLOR)
        text_rect = text.get_rect(center=(self.x + RECT_WIDTH // 2, self.y + RECT_HEIGHT // 2))
        window.blit(text, text_rect)

    def set_pos(self, ceil=False):
        if ceil:
            self.row = math.ceil(self.y / RECT_HEIGHT)
            self.col = math.ceil(self.x / RECT_WIDTH)
        else:
            self.row = math.floor(self.y / RECT_HEIGHT)
            self.col = math.floor(self.x / RECT_WIDTH)

    def move(self, delta):
        self.x += delta[0]
        self.y += delta[1]

def draw_grid(window):
    for row in range(1, ROWS):
        y = row * RECT_HEIGHT
        pygame.draw.line(window, OUTLINE_COLOR, (0, y), (WIDTH, y), OUTLINE_THICHKNESS)
    for col in range(1, COLS):
        x = col * RECT_WIDTH
        pygame.draw.line(window, OUTLINE_COLOR, (x, 0), (x, HEIGHT), OUTLINE_THICHKNESS)
    pygame.draw.rect(window, OUTLINE_COLOR, (0, 0, WIDTH, HEIGHT), OUTLINE_THICHKNESS)

def draw(window, tiles):
    window.fill(BACKGROUND_COLOR)
    for tile in tiles.values():
        tile.draw(window)
    draw_grid(window)
    pygame.display.update()

def get_random_pos(tiles):
    row = None
    col = None
    while True:
        row = random.randrange(0, ROWS)
        col = random.randrange(0, COLS)
        if f"{row}{col}" not in tiles:
            break
    return row, col

def move_tiles(window, tiles, clock, direction):
    updated = True
    blocks = set()
    old_positions = {k: (tile.value, tile.row, tile.col) for k, tile in tiles.items()}
    if direction == "left":
        sort_func = lambda x: x.col
        reverse = False
        delta = (-MOVE_VEL, 0)
        boundary_check = lambda tile: tile.col == 0
        get_next_tile = lambda tile: tiles.get(f"{tile.row}{tile.col - 1}", None)
        merge_check = lambda tile, next_tile: tile.x > next_tile.x + MOVE_VEL
        move_check = lambda tile, next_tile: tile.x > next_tile.x + RECT_WIDTH + MOVE_VEL
        ceil = True
    elif direction == "right":
        sort_func = lambda x: x.col
        reverse = True
        delta = (MOVE_VEL, 0)
        boundary_check = lambda tile: tile.col == COLS - 1
        get_next_tile = lambda tile: tiles.get(f"{tile.row}{tile.col + 1}", None)
        merge_check = lambda tile, next_tile: tile.x < next_tile.x - MOVE_VEL
        move_check = lambda tile, next_tile: tile.x < next_tile.x - RECT_WIDTH - MOVE_VEL
        ceil = False
    elif direction == "up":
        sort_func = lambda x: x.row
        reverse = False
        delta = (0, -MOVE_VEL)
        boundary_check = lambda tile: tile.row == 0
        get_next_tile = lambda tile: tiles.get(f"{tile.row - 1}{tile.col}", None)
        merge_check = lambda tile, next_tile: tile.y > next_tile.y + MOVE_VEL
        move_check = lambda tile, next_tile: tile.y > next_tile.y + RECT_HEIGHT + MOVE_VEL
        ceil = True
    elif direction == "down":
        sort_func = lambda x: x.row
        reverse = True
        delta = (0, MOVE_VEL)
        boundary_check = lambda tile: tile.row == ROWS - 1
        get_next_tile = lambda tile: tiles.get(f"{tile.row + 1}{tile.col}", None)
        merge_check = lambda tile, next_tile: tile.y < next_tile.y - MOVE_VEL
        move_check = lambda tile, next_tile: tile.y < next_tile.y - RECT_HEIGHT - MOVE_VEL
        ceil = False

    while updated:
        clock.tick(FPS)
        updated = False
        sorted_tiles = sorted(tiles.values(), key=sort_func, reverse=reverse)
        for i, tile in enumerate(sorted_tiles):
            if boundary_check(tile):
                continue
            next_tile = get_next_tile(tile)
            if not next_tile:
                tile.move(delta)
            elif tile.value == next_tile.value and next_tile not in blocks and tile not in blocks:
                if merge_check(tile, next_tile):
                    tile.move(delta)
                else:
                    next_tile.value *= 2
                    sorted_tiles.pop(i)
                    blocks.add(next_tile)
            elif move_check(tile, next_tile):
                tile.move(delta)
            else:
                continue
            tile.set_pos(ceil)
            updated = True
        updated_tiles(window, tiles, sorted_tiles)
    new_positions = {k: (tile.value, tile.row, tile.col) for k, tile in tiles.items()}
    if old_positions != new_positions or (old_positions == new_positions and len(tiles) == 16):
        return end_move(tiles)
    else:
        return "continue"

def end_move(tiles):
    if len(tiles) == 16:
        return "lost"
    row, col = get_random_pos(tiles)
    tiles[f"{row}{col}"] = Tile(random.choice([2, 2, 2, 2, 2, 2, 2, 2, 2, 4]), row, col)
    return "continue"

def updated_tiles(window, tiles, sorted_tiles):
    tiles.clear()
    for tile in sorted_tiles:
        tiles[f"{tile.row}{tile.col}"] = tile
    draw(window, tiles)

def generate_tile():
    tiles = {}
    for _ in range(2):
        row, col = get_random_pos(tiles)
        tiles[f"{row}{col}"] = Tile(2, row, col)
    return tiles

# Hàm heuristic
def count_empty_tiles(tiles):
    return 16 - len(tiles)

def max_in_corner(tiles):
    if not tiles:
        return 0
    max_tile = max(tile.value for tile in tiles.values())
    corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
    for row, col in corners:
        if f"{row}{col}" in tiles and tiles[f"{row}{col}"].value == max_tile:
            return 1
    return 0

def merge_potential(tiles):
    score = 0
    for tile in tiles.values():
        r, c = tile.row, tile.col
        for dr, dc in [(1, 0), (0, 1)]:
            adjacent_tile = tiles.get(f"{r + dr}{c + dc}")
            if adjacent_tile and adjacent_tile.value == tile.value:
                if tile.value > 128:  # Ưu tiên khối lớn hơn 128
                    score += tile.value * 2  # Tăng điểm số mạnh hơn cho khối lớn
                else:
                    score += math.log2(tile.value)
    return score

def high_value_merge(tiles):
    score = 0
    for tile in tiles.values():
        if tile.value >= 256:  # Chỉ xem xét các ô ≥ 256
            r, c = tile.row, tile.col
            for dr, dc in [(1, 0), (0, 1)]:
                adjacent_tile = tiles.get(f"{r + dr}{c + dc}")
                if adjacent_tile and adjacent_tile.value == tile.value:
                    score += tile.value  # Điểm số dựa trên giá trị ô
    return score

def smoothness(tiles):
    smooth = 0
    for tile in tiles.values():
        r, c = tile.row, tile.col
        for dr, dc in [(1, 0), (0, 1)]:
            neighbor = tiles.get(f"{r + dr}{c + dc}")
            if neighbor:
                smooth += abs(math.log2(tile.value) - math.log2(neighbor.value))
    return smooth

def monotonicity_row(tiles):
    score = 0
    for r in range(ROWS):
        for c in range(COLS-1):
            current_tile = tiles.get(f"{r}{c}")
            next_tile = tiles.get(f"{r}{c+1}")
            cur_val = current_tile.value if current_tile else 0
            next_val = next_tile.value if next_tile else 0
            if cur_val >= next_val:
                score += cur_val - next_val
            else:
                score -= next_val - cur_val
    return score

def monotonicity_col(tiles):
    score = 0
    for c in range(COLS):
        for r in range(ROWS-1):
            current_tile = tiles.get(f"{r}{c}")
            next_tile = tiles.get(f"{r+1}{c}")
            cur_val = current_tile.value if current_tile else 0
            next_val = next_tile.value if next_tile else 0
            if cur_val >= next_val:
                score += cur_val - next_val
            else:
                score -= next_val - cur_val
    return score

def monotonicity(tiles):
    return max(monotonicity_row(tiles), monotonicity_col(tiles))

def big_tile_grouping(tiles):
    n = 4
    score = 0
    for i in range(n):
        for j in range(n):
            tile = tiles.get(f"{i}{j}")
            if tile and tile.value >= 64:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < n and 0 <= nj < n:
                        neighbor_tile = tiles.get(f"{ni}{nj}")
                        if neighbor_tile and neighbor_tile.value >= 64:
                            score += math.log2(tile.value) + math.log2(neighbor_tile.value)
    return score

def total_tile_value(tiles):
    return sum(tile.value for tile in tiles.values())

def get_score(tiles):
    if not tiles:
        return 0
    
    # Chuẩn hóa các heuristic
    empty_tiles = count_empty_tiles(tiles) / 16  # [0,1]
    max_in_corner_score = max_in_corner(tiles)  # [0,1]
    merge_pot = merge_potential(tiles) / 10000  # Chuẩn hóa mạnh hơn do điểm số lớn
    high_merge = high_value_merge(tiles) / 10000  # Chuẩn hóa cho ô ≥ 256
    smooth = smoothness(tiles) / 100  # Ước lượng chuẩn hóa
    mono = monotonicity(tiles) / 10000  # Ước lượng chuẩn hóa
    big_tile = big_tile_grouping(tiles) / 200  # Ước lượng chuẩn hóa
    total_value = total_tile_value(tiles) / 10000  # Ước lượng chuẩn hóa

    # Tính điểm số với trọng số
    score = (
        empty_tiles * HEURISTIC_WEIGHTS['empty_tiles'] +
        max_in_corner_score * HEURISTIC_WEIGHTS['max_in_corner'] +
        merge_pot * HEURISTIC_WEIGHTS['merge_potential'] +
        high_merge * HEURISTIC_WEIGHTS['high_value_merge'] -
        smooth * HEURISTIC_WEIGHTS['smoothness'] +
        mono * HEURISTIC_WEIGHTS['monotonicity'] +
        big_tile * HEURISTIC_WEIGHTS['big_tile_group'] +
        total_value * HEURISTIC_WEIGHTS['total_tile_value']
    )
    return score

def is_same_board(tiles1, tiles2):
    if len(tiles1) != len(tiles2):
        return False
    for key in tiles1:
        if key not in tiles2:
            return False
        if tiles1[key].value != tiles2[key].value:
            return False
    return True

def simulate_move(tiles, direction):
    new_tiles = copy.deepcopy(tiles)
    if direction in ["left", "right"]:
        for r in range(ROWS):
            line = []
            for c in range(COLS):
                key = f"{r}{c}" if direction == "left" else f"{r}{COLS-1-c}"
                if key in new_tiles:
                    line.append(new_tiles[key])
            merged_line = []
            skip = False
            for i in range(len(line)):
                if skip:
                    skip = False
                    continue
                if i + 1 < len(line) and line[i].value == line[i + 1].value:
                    merged_tile = Tile(line[i].value * 2, r, 0)
                    merged_line.append(merged_tile)
                    skip = True
                else:
                    merged_tile = Tile(line[i].value, r, 0)
                    merged_line.append(merged_tile)
            for c in range(COLS):
                key = f"{r}{c}"
                new_tiles.pop(key, None)
            for i, tile in enumerate(merged_line):
                tile.col = i if direction == "left" else COLS - 1 - i
                new_tiles[f"{tile.row}{tile.col}"] = tile
    elif direction in ["up", "down"]:
        for c in range(COLS):
            line = []
            for r in range(ROWS):
                key = f"{r}{c}" if direction == "up" else f"{ROWS-1-r}{c}"
                if key in new_tiles:
                    line.append(new_tiles[key])
            merged_line = []
            skip = False
            for i in range(len(line)):
                if skip:
                    skip = False
                    continue
                if i + 1 < len(line) and line[i].value == line[i + 1].value:
                    merged_tile = Tile(line[i].value * 2, 0, c)
                    merged_line.append(merged_tile)
                    skip = True
                else:
                    merged_tile = Tile(line[i].value, 0, c)
                    merged_line.append(merged_tile)
            for r in range(ROWS):
                key = f"{r}{c}"
                new_tiles.pop(key, None)
            for i, tile in enumerate(merged_line):
                tile.row = i if direction == "up" else ROWS - 1 - i
                new_tiles[f"{tile.row}{tile.col}"] = tile
    return new_tiles

def available_cells(tiles):
    cells = []
    for r in range(ROWS):
        for c in range(COLS):
            if f"{r}{c}" not in tiles:
                cells.append((r, c))
    return cells

def expectimax(tiles, depth, agent):
    if depth == 0 or len(tiles) == 16:
        return get_score(tiles)

    if agent == PLAYER:
        max_score = -math.inf
        directions = ["left", "right", "up", "down"]
        for direction in directions:
            new_tiles = simulate_move(tiles, direction)
            if is_same_board(tiles, new_tiles):
                continue
            score = expectimax(new_tiles, depth - 1, BOARD)
            max_score = max(max_score, score)
        return max_score if max_score != -math.inf else -math.inf
    else:  # agent == BOARD
        score = 0
        cells = available_cells(tiles)
        if not cells:
            return get_score(tiles)
        total_cells = len(cells)
        for r, c in cells:
            new_tiles = copy.deepcopy(tiles)
            new_tiles[f"{r}{c}"] = Tile(2, r, c)
            score += 0.9 * expectimax(new_tiles, depth - 1, PLAYER)
            new_tiles = copy.deepcopy(tiles)
            new_tiles[f"{r}{c}"] = Tile(4, r, c)
            score += 0.1 * expectimax(new_tiles, depth - 1, PLAYER)
        return score / total_cells

def get_best_move(tiles, depth=MAX_DEPTH):
    directions = ["left", "right", "up", "down"]
    best_direction = None
    best_score = -math.inf
    for direction in directions:
        new_tiles = simulate_move(tiles, direction)
        if is_same_board(tiles, new_tiles):
            continue
        score = expectimax(new_tiles, depth - 1, BOARD)
        if score > best_score:
            best_score = score
            best_direction = direction
    return best_direction

def main(window):
    clock = pygame.time.Clock()
    tiles = generate_tile()
    run = True
    max_tile = 2

    while run:
        clock.tick(FPS)
        best_dir = get_best_move(tiles, MAX_DEPTH)
        if best_dir:
            result = move_tiles(window, tiles, clock, best_dir)
        else:
            result = "lost"

        if tiles:
            current_max = max(tile.value for tile in tiles.values())
            max_tile = max(max_tile, current_max)

        if result == "lost":
            run = False

        draw(window, tiles)

    total_value = sum(tile.value for tile in tiles.values()) if tiles else 0
    max_value = max_tile if tiles else 0
    reward = math.ceil(0.7 * total_value + 0.3 * math.log2(max_value) * 1000) if total_value > 0 and max_value > 0 else 0
    print("Reward:", reward, "MaxValue:", max_value)
    pygame.quit()

if __name__ == "__main__":
    main(WINDOW)