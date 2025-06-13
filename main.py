import pygame
import random
import math
import copy
import numpy as np
from collections import defaultdict
pygame.init()

FPS = 5000

WIDTH, HEIGHT = 800, 800
ROWS = 4
COLS = 4

RECT_HEIGHT = HEIGHT // ROWS
RECT_WIDTH = WIDTH // COLS

OUTLINE_COLOR = (120, 120, 120)
OUTLINE_THICHKNESS = 10
BACKGROUND_COLOR = (0, 0, 0)
FONT_COLOR = (255, 255, 0)

FONT = pygame.font.SysFont("comicsans", 70, bold = True)
MOVE_VEL = 20

WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2048")

class Tile: 
    COLORS = [
        (220, 220, 220),
        (192, 192, 192),
        (128, 128, 128),
        (105, 105, 105),
        (79, 79, 79),
        (54, 54, 54),
        (38, 38, 38),
        (0,0,0)
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

    def set_pos(self, ceil = False):
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
    for row in range(1,ROWS):
        y = row * RECT_HEIGHT
        pygame.draw.line(window, OUTLINE_COLOR, (0, y), (WIDTH, y), OUTLINE_THICHKNESS)
    for col in range(1,COLS):
        x = col * RECT_WIDTH
        pygame.draw.line(window, OUTLINE_COLOR, (x, 0), (x, HEIGHT), OUTLINE_THICHKNESS)
    pygame.draw.rect(window, OUTLINE_COLOR, (0,0,WIDTH, HEIGHT), OUTLINE_THICHKNESS)

def draw(window,tiles):
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
        sort_func = lambda x : x.col
        reverse = False
        delta = (-MOVE_VEL, 0)
        boundary_check = lambda tile: tile.col == 0
        get_next_tile = lambda tile: tiles.get(f"{tile.row}{tile.col - 1}", None)
        merge_check = lambda tile, next_tile: tile.x > next_tile.x + MOVE_VEL
        move_check = (lambda tile, next_tile: tile.x > next_tile.x + RECT_WIDTH + MOVE_VEL)
        ceil = True 
    elif direction == "right":
        sort_func = lambda x : x.col
        reverse = True
        delta = (MOVE_VEL, 0)
        boundary_check = lambda tile: tile.col == COLS - 1
        get_next_tile = lambda tile: tiles.get(f"{tile.row}{tile.col + 1}", None)
        merge_check = lambda tile, next_tile: tile.x < next_tile.x - MOVE_VEL
        move_check = (lambda tile, next_tile: tile.x < next_tile.x - RECT_WIDTH - MOVE_VEL)
        ceil = False
    elif direction == "up":
        sort_func = lambda x : x.row
        reverse = False
        delta = (0, -MOVE_VEL)
        boundary_check = lambda tile: tile.row == 0
        get_next_tile = lambda tile: tiles.get(f"{tile.row - 1}{tile.col}", None)
        merge_check = lambda tile, next_tile: tile.y > next_tile.y + MOVE_VEL
        move_check = (lambda tile, next_tile: tile.y > next_tile.y + RECT_HEIGHT + MOVE_VEL)
        ceil = True
    elif direction == "down":
        sort_func = lambda x : x.row
        reverse = True
        delta = (0, MOVE_VEL)
        boundary_check = lambda tile: tile.row == ROWS - 1
        get_next_tile = lambda tile: tiles.get(f"{tile.row + 1}{tile.col}", None)
        merge_check = lambda tile, next_tile: tile.y < next_tile.y - MOVE_VEL
        move_check = (lambda tile, next_tile: tile.y < next_tile.y - RECT_HEIGHT - MOVE_VEL)
        ceil = False

    while updated:
        clock.tick(FPS)
        updated = False
        sorted_tiles = sorted(tiles.values(), key=sort_func, reverse=reverse)

        for i,tile in enumerate(sorted_tiles):
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
    tiles[f"{row}{col}"] = Tile(random.choice([2, 4]), row, col)
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


############ heuristic functions ############
def is_same_board(tiles1, tiles2):
    if len(tiles1) != len(tiles2):
        return False
    for key in tiles1:
        if key not in tiles2:
            return False
        if tiles1[key].value != tiles2[key].value:
            return False
    return True
def count_empty_tiles(tiles):
    return 16 - len(tiles)
def max_tile_in_corner(tiles):
    if not tiles:
        return 0
    max_tile = max(tile.value for tile in tiles.values())
    corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
    for row, col in corners:
        if f"{row}{col}" in tiles and tiles[f"{row}{col}"].value == max_tile:
            return 1
    return 0
def count_adjacent_tiles(tiles):
    count = 0
    for tile in tiles.values():
        r, c = tile.row, tile.col
        for dr, dc in [(1, 0), (0, 1)]:
            adjacent_tile = tiles.get(f"{r + dr}{c + dc}")
            if adjacent_tile and adjacent_tile.value == tile.value:
                count += 1
    return count
def smoothness(tiles):
    smooth = 0
    for tile in tiles.values():
        r, c = tile.row, tile.col
        for dr, dc in [(1, 0), (0, 1)]:
            neighbor = tiles.get(f"{r + dr}{c + dc}")
            if neighbor:
                smooth += abs(tile.value - neighbor.value)
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
def big_tile_grouping(tiles):
    n = 4
    score = 0
    for i in range(n):
        for j in range(n):
            tile = tiles.get(f"{i}{j}")
            if tile and tile.value >= 64:
                for dx, dy in [(-1,0), (1,0)]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < n and 0 <= nj < n:
                        neighbor_tile = tiles.get(f"{ni}{nj}")
                        if neighbor_tile:
                            if   neighbor_tile.value >= 64:
                                score += math.log2(tile.value) + math.log2(neighbor_tile.value)
    return score
def normalized_smoothness(tiles):
    smooth = smoothness(tiles)
    max_smooth = 10000  # Ước lượng giá trị tối đa của smoothness
    return smooth / max_smooth if max_smooth > 0 else 0
def monotonicity(tiles):
    score = max(monotonicity_row(tiles), monotonicity_col(tiles))
    return score
def distance_to_corner(tiles):
    if not tiles:
        return 0
    max_tile = max(tiles.values(), key=lambda t: t.value)
    corners = [(0,0), (0,3), (3,0), (3,3)]
    min_distance = min(abs(max_tile.row - r) + abs(max_tile.col - c) for r, c in corners)
    return 1 / (1 + min_distance)
def merge_potential(tiles):
    score = 0
    for tile in tiles.values():
        r, c = tile.row, tile.col
        for dr, dc in [(1, 0), (0, 1)]:
            adjacent_tile = tiles.get(f"{r + dr}{c + dc}")
            if adjacent_tile and adjacent_tile.value == tile.value:
                score += math.log2(tile.value)  # Ưu tiên gộp ô lớn
    return score
def total_tile_value(tiles):
    return sum(tile.value for tile in tiles.values()) / 10000
def heuristic_score(tiles, weights):
    empty_tiles = count_empty_tiles(tiles) / 16  # Chuẩn hóa về [0,1]
    distance_corner = distance_to_corner(tiles)  # Thay max_tile_in_corner
    merge_pot = merge_potential(tiles) / 20  # Ước lượng giá trị tối đa
    norm_smooth = normalized_smoothness(tiles)  # smoothness / 10000
    mono = monotonicity(tiles) / 10000  # Chuẩn hóa
    big_tile = big_tile_grouping(tiles) / 200  # Ước lượng giá trị tối đa
    total_value = total_tile_value(tiles)  # sum(tile.value) / 10000
    
    score = (
        empty_tiles * weights['empty_tiles']
        + distance_corner * weights['distance_to_corner']
        + merge_pot * weights['merge_potential']
        - norm_smooth * weights['smoothness']
        + mono * weights['monotonicity']
        + big_tile * weights['big_tile_group']
        + total_value * weights['total_tile_value']
    )
    return score

def simulate_move(tiles, direction):
    """
    Trả về một board tiles mới sau khi thực hiện thử 1 nước đi.
    Không sinh tile mới.
    """
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
                if i+1 < len(line) and line[i].value == line[i+1].value:
                    merged_tile = Tile(line[i].value * 2, r, 0)  # col sẽ set sau
                    merged_line.append(merged_tile)
                    skip = True
                else:
                    merged_tile = Tile(line[i].value, r, 0)
                    merged_line.append(merged_tile)

            for c in range(COLS):
                key = f"{r}{c}"
                new_tiles.pop(key, None)

            for i, tile in enumerate(merged_line):
                tile.col = i if direction == "left" else COLS-1-i
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
                if i+1 < len(line) and line[i].value == line[i+1].value:
                    merged_tile = Tile(line[i].value * 2, 0, c)  # row sẽ set sau
                    merged_line.append(merged_tile)
                    skip = True
                else:
                    merged_tile = Tile(line[i].value, 0, c)
                    merged_line.append(merged_tile)

            for r in range(ROWS):
                key = f"{r}{c}"
                new_tiles.pop(key, None)

            for i, tile in enumerate(merged_line):
                tile.row = i if direction == "up" else ROWS-1-i
                new_tiles[f"{tile.row}{tile.col}"] = tile

    return new_tiles
def best_move(tiles, weights):
    directions = ["left", "right", "up", "down"]
    best_dir = None
    best_score = -math.inf

    for direction in directions:
        new_tiles = simulate_move(tiles, direction)
        if is_same_board(tiles, new_tiles):
            continue
        score = heuristic_score(new_tiles, weights)
        if score > best_score:
            best_score = score
            best_dir = direction
    
    return best_dir

def play_game(window, clock, weights):
    tiles = generate_tile()
    run = True
    max_tile = 2  # Theo dõi ô lớn nhất

    while run:
        clock.tick(FPS)
        best_dir = best_move(tiles, weights)
        if best_dir:
            result = move_tiles(window, tiles, clock, best_dir)
        else:
            result = "lost"
        
        # Cập nhật max_tile
        if tiles:
            current_max = max(tile.value for tile in tiles.values())
            max_tile = max(max_tile, current_max)
        
        if result == "lost":
            run = False
        
        draw(window, tiles)
    
    # Phần thưởng kết hợp: 70% từ tổng giá trị các ô, 30% từ log2 của ô lớn nhất
    total_value = sum(tile.value for tile in tiles.values()) if tiles else 0
    max_value = max_tile if tiles else 0
    reward = math.ceil(0.7 * total_value + 0.3 * math.log2(max_value) * 1000) if total_value > 0 and max_value > 0 else 0
    print("Reward: ", reward, "MaxValue: ", max_value)
    return reward

def q_learning_optimize_weights():
    # Khởi tạo trọng số ban đầu
    weights = {
            'empty_tiles': 500.0,
            'distance_to_corner': 1000.0,
            'merge_potential': 200.0,
            'smoothness': 300.0,
            'monotonicity': 400.0,
            'big_tile_group': 600.0,
            'total_tile_value': 500.0
        }
    
    # Các tham số Q-Learning
    alpha = 0.1  # Tốc độ học
    gamma = 0.9  # Hệ số chiết khấu
    epsilon = 0.5  # Xác suất exploration
    episodes = 1000  # Số ván chơi để huấn luyện
    
    # Q-Table: ánh xạ (state, action) -> Q-value
    Q = defaultdict(lambda: 0.0)
    
    # Các hành động: tăng hoặc giảm mỗi trọng số
    weight_keys = list(weights.keys())
    actions = []
    for key in weight_keys:
        actions.append((key, 0.1 * weights[key]))  # Tăng 10%
        actions.append((key, -0.1 * weights[key])) # Giảm 10%
    
    # Discretize trạng thái để giảm kích thước Q-Table
    def discretize_weights(weights):
        return tuple(round(w, 1) for w in weights.values())

    # Chạy Q-Learning
    for episode in range(episodes):
        current_weights = weights.copy()
        state = discretize_weights(current_weights)
        
        # Chơi một ván trò chơi
        clock = pygame.time.Clock()
        reward = play_game(WINDOW, clock, current_weights)
        
        # Chọn hành động
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            # Chọn hành động có Q-value cao nhất
            action = max(actions, key=lambda a: Q[(state, a)], default=actions[0])
        
        # Thực hiện hành động
        weight_key, change = action
        new_weights = current_weights.copy()
        new_weights[weight_key] += change
        # Giới hạn trọng số để tránh giá trị không hợp lý
        new_weights[weight_key] = max(0.0, min(new_weights[weight_key], 1000000.0))
        
        # Trạng thái mới
        new_state = discretize_weights(new_weights)
        
        # Cập nhật Q-Table
        best_future_q = max(Q[(new_state, a)] for a in actions) if new_state in [k[0] for k in Q.keys()] else 0
        Q[(state, action)] = Q[(state, action)] + alpha * (reward + gamma * best_future_q - Q[(state, action)])
        
        # Cập nhật trọng số
        weights = new_weights
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {reward}, Weights: {weights}")
    
    pygame.quit()
    return weights






# def main(window):
#     # Tối ưu hóa trọng số bằng Q-Learning
#     optimal_weights = q_learning_optimize_weights()
#     print("Optimal weights:", optimal_weights)
    
#     # Chạy trò chơi với trọng số tối ưu (tùy chọn)
#     clock = pygame.time.Clock()
#     tiles = generate_tile()
#     run = True
    
#     while run:
#         clock.tick(FPS)
#         best_dir = best_move(tiles, optimal_weights)
#         if best_dir:
#             result = move_tiles(window, tiles, clock, best_dir)
#         else:
#             result = "lost"
        
#         if result == "lost":
#             print("Game Over!")
#             run = False
        
#         draw(window, tiles)
    
#     pygame.quit()

# if __name__ == "__main__":
#     main(WINDOW)


###################################################################################
def main(window):
    clock = pygame.time.Clock()
    run = True

    # tiles = generate_tile()
    # while run:
    #     clock.tick(FPS)
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             run = False
    #             break
    #         if event.type == pygame.KEYDOWN:
    #             if event.key == pygame.K_LEFT:
    #                 result = move_tiles(window, tiles, clock, "left")
    #             elif event.key == pygame.K_RIGHT:
    #                 result = move_tiles(window, tiles, clock, "right")
    #             elif event.key == pygame.K_UP:
    #                 result = move_tiles(window, tiles, clock, "up")
    #             elif event.key == pygame.K_DOWN:
    #                 result = move_tiles(window, tiles, clock, "down")
                
    #             if result == "lost":
    #                 print("Game Over!")
    #                 run = False
    #                 break
    #     draw(window, tiles)
    tiles = generate_tile()
    run = True
    max_tile = 2  # Theo dõi ô lớn nhất

    while run:
        clock.tick(FPS)
        best_dir = best_move(tiles, {
            'empty_tiles': 23300.0,
            'distance_to_corner': 1200.0,
            'merge_potential': 300.0,
            'smoothness': 390.0,
            'monotonicity': 480.0,
            'big_tile_group': 1140.0,
            'total_tile_value': 450.0
        })
        if best_dir:
            result = move_tiles(window, tiles, clock, best_dir)
        else:
            result = "lost"
        
        # Cập nhật max_tile
        if tiles:
            current_max = max(tile.value for tile in tiles.values())
            max_tile = max(max_tile, current_max)
        
        if result == "lost":
            run = False
        
        draw(window, tiles)
    
    # Phần thưởng kết hợp: 70% từ tổng giá trị các ô, 30% từ log2 của ô lớn nhất
    total_value = sum(tile.value for tile in tiles.values()) if tiles else 0
    max_value = max_tile if tiles else 0
    reward = math.ceil(0.7 * total_value + 0.3 * math.log2(max_value) * 1000) if total_value > 0 and max_value > 0 else 0
    print("Reward: ", reward, "MaxValue: ", max_value)
    
    pygame.quit()

if __name__ == "__main__":
    main(WINDOW)