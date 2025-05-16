import pygame
import random
import sys
import cv2
import mediapipe as mp
import numpy as np

# 初始化pygame
pygame.init()

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)

# 手势识别初始化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 游戏设置
GRID_SIZE = 30
GRID_WIDTH = 9
GRID_HEIGHT = 13
GAME_AREA_WIDTH = GRID_WIDTH * GRID_SIZE
GAME_AREA_HEIGHT = GRID_HEIGHT * GRID_SIZE
SIDEBAR_WIDTH = 300
SCREEN_WIDTH = GAME_AREA_WIDTH + SIDEBAR_WIDTH
SCREEN_HEIGHT = GAME_AREA_HEIGHT

# 方块形状
SHAPES = [
    [[1, 1, 1, 1]],  # I
    [[1, 1], [1, 1]],  # O
    [[1, 1, 1], [0, 1, 0]],  # T
    [[1, 1, 1], [1, 0, 0]],  # L
    [[1, 1, 1], [0, 0, 1]],  # J
    [[0, 1, 1], [1, 1, 0]],  # S
    [[1, 1, 0], [0, 1, 1]]   # Z
]

# 方块颜色
COLORS = [CYAN, YELLOW, MAGENTA, ORANGE, BLUE, GREEN, RED]

# 创建游戏窗口
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("俄罗斯方块")

# 游戏区域
game_area = pygame.Rect(0, 0, GAME_AREA_WIDTH, GAME_AREA_HEIGHT)
# 辅助区域
sidebar_area = pygame.Rect(GAME_AREA_WIDTH, 0, SIDEBAR_WIDTH, SCREEN_HEIGHT)

# 游戏网格
grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

# 当前方块
current_piece = None
current_x = 0
current_y = 0
current_color = None

# 游戏时钟
clock = pygame.time.Clock()
fall_time = 0
fall_speed = 1000  # 1秒下落一次

# 游戏状态
game_over = False

def draw_grid():
    # 绘制游戏区域背景
    pygame.draw.rect(screen, GRAY, game_area, 0)
    # 绘制网格线
    for x in range(GRID_WIDTH + 1):
        pygame.draw.line(screen, BLACK, (x * GRID_SIZE, 0), (x * GRID_SIZE, GAME_AREA_HEIGHT), 1)
    for y in range(GRID_HEIGHT + 1):
        pygame.draw.line(screen, BLACK, (0, y * GRID_SIZE), (GAME_AREA_WIDTH, y * GRID_SIZE), 1)

def draw_sidebar(camera_image=None, direction_text=""):
    # 绘制辅助区域背景
    pygame.draw.rect(screen, WHITE, sidebar_area, 0)
    # 绘制辅助区域边框
    pygame.draw.rect(screen, BLACK, sidebar_area, 1)
    
    if camera_image is not None:
        # 调整摄像头图像大小以适应侧边栏
        camera_image = cv2.resize(camera_image, (SIDEBAR_WIDTH, SCREEN_HEIGHT - 100))
        # 将OpenCV图像转换为Pygame表面
        camera_surface = pygame.surfarray.make_surface(camera_image.swapaxes(0, 1))
        screen.blit(camera_surface, (GAME_AREA_WIDTH, 0))
        
        # 显示方向文本
        font = pygame.font.SysFont(None, 36)
        text = font.render(f"Direction: {direction_text}", True, BLACK)
        screen.blit(text, (GAME_AREA_WIDTH + 10, SCREEN_HEIGHT - 80))

def new_piece():
    global current_piece, current_x, current_y, current_color
    shape_idx = random.randint(0, len(SHAPES) - 1)
    current_piece = SHAPES[shape_idx]
    current_color = COLORS[shape_idx]
    current_x = GRID_WIDTH // 2 - len(current_piece[0]) // 2
    current_y = 0
    
    # 检查游戏是否结束
    if check_collision():
        return False
    return True

def draw_piece():
    for y, row in enumerate(current_piece):
        for x, cell in enumerate(row):
            if cell:
                pygame.draw.rect(screen, current_color, 
                                (current_x * GRID_SIZE + x * GRID_SIZE, 
                                 current_y * GRID_SIZE + y * GRID_SIZE, 
                                 GRID_SIZE, GRID_SIZE), 0)
                pygame.draw.rect(screen, BLACK, 
                                (current_x * GRID_SIZE + x * GRID_SIZE, 
                                 current_y * GRID_SIZE + y * GRID_SIZE, 
                                 GRID_SIZE, GRID_SIZE), 1)

def rotate_piece():
    global current_piece
    # 转置矩阵并反转每一行实现旋转
    rotated = list(zip(*current_piece[::-1]))
    current_piece = [list(row) for row in rotated]

def check_collision(dx=0, dy=0):
    for y, row in enumerate(current_piece):
        for x, cell in enumerate(row):
            if cell:
                new_x = current_x + x + dx
                new_y = current_y + y + dy
                if (new_x < 0 or new_x >= GRID_WIDTH or 
                    new_y >= GRID_HEIGHT or 
                    (new_y >= 0 and grid[new_y][new_x])):
                    return True
    return False

def merge_piece():
    for y, row in enumerate(current_piece):
        for x, cell in enumerate(row):
            if cell and current_y + y >= 0:
                grid[current_y + y][current_x + x] = current_color

def clear_lines():
    global grid
    lines_cleared = 0
    for y in range(GRID_HEIGHT):
        if all(grid[y]):
            lines_cleared += 1
            for y2 in range(y, 0, -1):
                grid[y2] = grid[y2-1][:]
            grid[0] = [0] * GRID_WIDTH
    return lines_cleared

def draw_grid_pieces():
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if grid[y][x]:
                pygame.draw.rect(screen, grid[y][x], 
                                (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE), 0)
                pygame.draw.rect(screen, BLACK, 
                                (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE), 1)

def show_game_over():
    font = pygame.font.SysFont(None, 48)
    text = font.render("Game Over", True, RED)
    text_rect = text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 30))
    
    button_font = pygame.font.SysFont(None, 36)
    button_text = button_font.render("Restart", True, BLACK)
    button_rect = pygame.Rect(SCREEN_WIDTH//2 - 50, SCREEN_HEIGHT//2 + 20, 100, 40)
    
    screen.fill(WHITE)
    screen.blit(text, text_rect)
    pygame.draw.rect(screen, GREEN, button_rect, 0)
    pygame.draw.rect(screen, BLACK, button_rect, 1)
    screen.blit(button_text, (button_rect.x + 20, button_rect.y + 10))
    
    pygame.display.flip()
    
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                hands.close()
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    waiting = False
                    return True
    return False

def reset_game():
    global grid, game_over
    grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    game_over = False
    new_piece()

# 游戏主循环
def process_gesture(hand_landmarks):
    # 获取食指和中指的指尖坐标
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    
    # 计算方向向量
    dx = index_tip.x - index_mcp.x
    dy = index_tip.y - index_mcp.y
    
    # 根据方向判断手势
    if abs(dx) > abs(dy):
        return "LEFT" if dx < 0 else "RIGHT"
    else:
        return "UP" if dy < 0 else "DOWN"

def main():
    global fall_time, game_over, current_x, current_y
    
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 手势操作间隔控制
    last_gesture_time = 0
    gesture_interval = 500  # 500毫秒间隔
    
    if not new_piece():
        game_over = True
    
    running = True
    while running:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if not game_over:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT and not check_collision(-1, 0):
                        current_x -= 1
                    elif event.key == pygame.K_RIGHT and not check_collision(1, 0):
                        current_x += 1
                    elif event.key == pygame.K_DOWN and not check_collision(0, 1):
                        current_y += 1
                    elif event.key == pygame.K_UP:
                        rotate_piece()
                        if check_collision():
                            # 如果旋转后发生碰撞，则撤销旋转
                            for _ in range(3):
                                rotate_piece()
        
        # 游戏逻辑
        if not game_over:
            # 方块自动下落
            fall_time += clock.get_rawtime()
            clock.tick()
            
            if fall_time >= fall_speed:
                fall_time = 0
                if not check_collision(0, 1):
                    current_y += 1
                else:
                    merge_piece()
                    clear_lines()
                    if not new_piece():
                        game_over = True
        
        # 处理摄像头输入
        ret, frame = cap.read()
        direction_text = ""
        if ret:
            # 转换颜色空间并水平翻转
            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            
            # 绘制手势标记
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    direction_text = process_gesture(hand_landmarks)
                    
                    # 根据手势控制方块(带间隔控制)
                    current_time = pygame.time.get_ticks()
                    if not game_over and current_time - last_gesture_time > gesture_interval:
                        if direction_text == "LEFT" and not check_collision(-1, 0):
                            current_x -= 1
                            last_gesture_time = current_time
                        elif direction_text == "RIGHT" and not check_collision(1, 0):
                            current_x += 1
                            last_gesture_time = current_time
                        elif direction_text == "DOWN" and not check_collision(0, 1):
                            current_y += 1
                            last_gesture_time = current_time
                        elif direction_text == "UP":
                            rotate_piece()
                            if check_collision():
                                for _ in range(3):
                                    rotate_piece()
                            last_gesture_time = current_time
            
            # 转换回BGR用于显示
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 绘制
        screen.fill(BLACK)
        draw_grid()
        draw_sidebar(image, direction_text)
        draw_grid_pieces()
        if not game_over:
            draw_piece()
        
        pygame.display.flip()
        
        # 游戏结束处理
        if game_over:
            if show_game_over():
                reset_game()
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
