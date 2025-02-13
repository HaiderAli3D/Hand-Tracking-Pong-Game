import pygame
import sys
import cv2
import mediapipe as mp
import numpy as np
import mediapipe_cheats as mpc

# Game constants
WIDTH, HEIGHT = 1400, 1200
PADDLE_WIDTH, PADDLE_HEIGHT = 20, 200
BALL_SIZE = 20
PADDLE_SPEED = 8
BALL_SPEED = 1000  #
MAX_BALL_SPEED = 1200  
HAND_SENSITIVITY = 2  
WALL_THICKNESS = 35 

# Control area constants
CONTROL_AREA_PERCENTAGE = 80
margin_percentage = (100 - CONTROL_AREA_PERCENTAGE) / 2
margin = margin_percentage / 100

# control area boundaries
control_area = {
    "x_min": margin,          
    "x_max": 1 - margin,     
    "y_min": margin,          
    "y_max": 1 - margin       
}

class PongGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Single Player Pong")
        self.clock = pygame.time.Clock()

        self.left_paddle = pygame.Rect(10, HEIGHT//2 - PADDLE_HEIGHT//2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.left_paddle_colli = pygame.Rect(-10, HEIGHT//2 - PADDLE_HEIGHT//2, PADDLE_WIDTH + 20, PADDLE_HEIGHT)
        self.wall = pygame.Rect(WIDTH - WALL_THICKNESS, 0, WALL_THICKNESS, HEIGHT)
        self.ball = pygame.Rect(WIDTH//2 - BALL_SIZE//2, HEIGHT//2 - BALL_SIZE//2, BALL_SIZE, BALL_SIZE)
        
        self.ball_direction = pygame.math.Vector2()
        self.ball_speed = BALL_SPEED
        self.reset_ball()
        
        self.score = 0
        self.font = pygame.font.Font(None, 36)

    def move_paddle(self, delta):
        # Move left paddle
        self.left_paddle.y += delta
        if self.left_paddle.top < 0:
            self.left_paddle.top = 0
        if self.left_paddle.bottom > HEIGHT:
            self.left_paddle.bottom = HEIGHT
        
        self.left_paddle_colli.top = self.left_paddle.top
        self.left_paddle_colli.bottom = self.left_paddle.bottom
        self.left_paddle_colli.height = self.left_paddle.height


    def update_ball(self, dt):
        # Update ball position
        self.ball.x += self.ball_direction.x * self.ball_speed * dt
        self.ball.y += self.ball_direction.y * self.ball_speed * dt

        # Ball collision with wall - bounce when reaching a certain x position
        if self.ball.x >= WIDTH - WALL_THICKNESS - BALL_SIZE:
            self.ball.x = WIDTH - WALL_THICKNESS - BALL_SIZE  # Place ball just before wall
            self.ball_direction.x *= -1
            self.ball_speed = min(self.ball_speed * 1.05, MAX_BALL_SPEED)

        # Ball col with top and bottom walls
        if self.ball.bottom <= 0:
            self.ball.y = 10
            self.ball_direction.y *= -1
        
        if self.ball.top >= HEIGHT:
            self.ball.y = HEIGHT - 1
            self.ball_direction.y *= -1
        

        # Ball col  with paddle
        if (self.ball.colliderect(self.left_paddle_colli) or 
            (self.ball.x < 0 and (self.ball.y < self.left_paddle.top and self.ball.y > self.left_paddle.bottom) )):
            self.score += 1
            # Calculate relative col point (-1 to 1)
            relative_intersect_y = (self.left_paddle_colli.centery - self.ball.centery) / (PADDLE_HEIGHT / 2)
            
            # Bounce angle (maximum 75 degrees in radians)
            bounce_angle = relative_intersect_y * 0.75
            
            # Set new direction while maintaining ball speed
            self.ball_direction = pygame.math.Vector2(
                np.cos(bounce_angle),
                -np.sin(bounce_angle)
            ).normalize()
            
            # Increase ball speed but cap it
            self.ball_speed = min(self.ball_speed * 1.05, MAX_BALL_SPEED)

        # Ball out of bounds (left side only)
        if self.ball.x <= -15:
            self.reset_ball()

    def reset_ball(self):
        self.score = 0
        
        # Reset ball to center and ball speed to base speed
        self.ball.center = (WIDTH // 2, HEIGHT // 2)
        self.ball_speed = BALL_SPEED
        
        # Random angle between -45 and 45 degrees (in radians)
        angle = np.random.uniform(-np.pi/4, np.pi/4)
        # Always start towards the wall
        self.ball_direction = pygame.math.Vector2(np.cos(angle), np.sin(angle)).normalize()

    def draw(self):
        # Clear the screen
        self.screen.fill((0, 0, 0))
        
        # Draw paddle, wall and ball
        pygame.draw.rect(self.screen, (255, 255, 255), self.left_paddle)
        pygame.draw.rect(self.screen, (255, 255, 255), self.wall)
        pygame.draw.ellipse(self.screen, (255, 255, 255), self.ball)
        
        # Draw score
        score_text = f"Score: {self.score}"
        text_surface = self.font.render(score_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(WIDTH // 2, 30))
        self.screen.blit(text_surface, text_rect)
        
        pygame.display.flip()

def main():
    #Init MediaPip Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,  # Only need to track one hand now
        min_detection_confidence=0.8,
        min_tracking_confidence=0.7,
    )
    mp_drawing = mp.solutions.drawing_utils

    #Init camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot initiate camera")
        sys.exit()

    game = PongGame()
    running = True

    while running:
        # delta time in seconds
        dt = game.clock.tick(60) / 1000.0

        # Process Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        paddle_delta = 0

        # Process hand tracking
        success, frame = cap.read()
        if not success:
            print("Failed to capture camera frame")
            break

        #Flip frame 
        frame = cv2.flip(frame, 1)
        
        #Convert BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hands in the frame
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Only need first hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            hand_x, hand_y = mpc.get_hand_center(hand_landmarks)
            
            # ensure hand position is in control area
            hand_y = max(control_area["y_min"], min(control_area["y_max"], hand_y))
            
            # Calculate relative position within control area (0-1)
            relative_y = (hand_y - control_area["y_min"]) / (control_area["y_max"] - control_area["y_min"])
            
            # Map to game coordinates
            paddle_y = relative_y * HEIGHT
            
            # Set paddle position
            target_y = paddle_y - PADDLE_HEIGHT / 2
            current_y = game.left_paddle.y
            paddle_delta = (target_y - current_y) * HAND_SENSITIVITY * 0.5

        # Draw control area rectangle
        frame_height, frame_width = frame.shape[:2]
        x1 = int(control_area["x_min"] * frame_width)
        y1 = int(control_area["y_min"] * frame_height)
        x2 = int(control_area["x_max"] * frame_width)
        y2 = int(control_area["y_max"] * frame_height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Show the camera feed
        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == 27:  
            break

        # Update game state and render
        game.move_paddle(paddle_delta)
        game.update_ball(dt)
        game.draw()

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
