import pygame
import sys
import cv2
import mediapipe as mp
import numpy as np
import mediapipe_cheats as mpc

# Game constants
WIDTH, HEIGHT = 1200, 900
PADDLE_WIDTH, PADDLE_HEIGHT = 20, 180
BALL_SIZE = 20
PADDLE_SPEED = 8
BALL_SPEED = 500  # Base speed in pixels per second
HAND_SENSITIVITY = 2  # Increase to make paddles more responsive

# Control area constants
CONTROL_AREA_PERCENTAGE = 80
margin_percentage = (100 - CONTROL_AREA_PERCENTAGE) / 2
margin = margin_percentage / 100

# Define the control area boundaries
control_area = {
    "x_min": margin,          # e.g., 0.10 for 80% width
    "x_max": 1 - margin,      # e.g., 0.90 for 80% width
    "y_min": margin,          # e.g., 0.10 for 80% height
    "y_max": 1 - margin       # e.g., 0.90 for 80% height
}

def map_to_screen_coordinates(x, y):
    # Ensure the point is within the control area
    x = max(control_area["x_min"], min(control_area["x_max"], x))
    y = max(control_area["y_min"], min(control_area["y_max"], y))
    
    # Map from control area to screen coordinates
    x_normalized = (x - control_area["x_min"]) / (control_area["x_max"] - control_area["x_min"])
    y_normalized = (y - control_area["y_min"]) / (control_area["y_max"] - control_area["y_min"])
    
    # Convert to screen coordinates
    screen_x = x_normalized * WIDTH
    screen_y = y_normalized * HEIGHT
    
    return screen_x, screen_y

class PongGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Hand Controlled Pong")
        self.clock = pygame.time.Clock()

        # Define paddles and ball
        self.left_paddle = pygame.Rect(10, HEIGHT//2 - PADDLE_HEIGHT//2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.right_paddle = pygame.Rect(WIDTH - 10 - PADDLE_WIDTH, HEIGHT//2 - PADDLE_HEIGHT//2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.ball = pygame.Rect(WIDTH//2 - BALL_SIZE//2, HEIGHT//2 - BALL_SIZE//2, BALL_SIZE, BALL_SIZE)
        
        # Initialize ball direction and speed
        self.ball_direction = pygame.math.Vector2()
        self.ball_speed = BALL_SPEED
        self.reset_ball()
        
        # Scores and font
        self.left_score = 0
        self.right_score = 0
        self.font = pygame.font.Font(None, 36)

    def move_paddles(self, left_delta, right_delta):
        # Move left paddle
        self.left_paddle.y += left_delta
        if self.left_paddle.top < 0:
            self.left_paddle.top = 0
        if self.left_paddle.bottom > HEIGHT:
            self.left_paddle.bottom = HEIGHT

        # Move right paddle
        self.right_paddle.y += right_delta
        if self.right_paddle.top < 0:
            self.right_paddle.top = 0
        if self.right_paddle.bottom > HEIGHT:
            self.right_paddle.bottom = HEIGHT

    def update_ball(self, dt):
        
        
        # Update ball position using delta time for frame-rate independence
        self.ball.x += self.ball_direction.x * self.ball_speed * dt
        self.ball.y += self.ball_direction.y * self.ball_speed * dt
    
        # Ball collision with top and bottom walls
        if self.ball.top <= 0 or self.ball.bottom >= HEIGHT:
            self.ball_direction.y *= -1

        # Ball collision with paddles
        if self.ball.colliderect(self.left_paddle) or self.ball.colliderect(self.right_paddle):
            paddle = self.left_paddle if self.ball.colliderect(self.left_paddle) else self.right_paddle
            
            # Calculate relative intersection point (-1 to 1)
            relative_intersect_y = (paddle.centery - self.ball.centery) / (PADDLE_HEIGHT / 2)
            
            # Bounce angle (maximum 75 degrees in radians)
            bounce_angle = relative_intersect_y * 0.75
            
            # Set new direction while maintaining ball speed
            if self.ball.colliderect(self.left_paddle):
                self.ball_direction = pygame.math.Vector2(
                    np.cos(bounce_angle),
                    -np.sin(bounce_angle)
                )
            else:
                self.ball_direction = pygame.math.Vector2(
                    -np.cos(bounce_angle),
                    -np.sin(bounce_angle)
                )
            self.ball_direction = self.ball_direction.normalize()
            # Increase ball speed by 5% after a paddle hit
            self.ball_speed *= 1.05

        # Ball out of bounds: update score and reset the ball
        if self.ball.left <= 0:
            self.right_score += 1
            self.reset_ball()
        elif self.ball.right >= WIDTH:
            self.left_score += 1
            self.reset_ball()

    def reset_ball(self):
        # Reset ball to center and ball speed to base speed
        self.ball.center = (WIDTH // 2, HEIGHT // 2)
        self.ball_speed = BALL_SPEED
        
        # Random angle between -45 and 45 degrees (in radians)
        angle = np.random.uniform(-np.pi/4, np.pi/4)
        # 50% chance to start towards either player
        if np.random.random() < 0.5:
            angle += np.pi
            
        # Set and normalize direction vector
        self.ball_direction = pygame.math.Vector2(np.cos(angle), np.sin(angle)).normalize()

    def draw(self):
        # Clear the screen
        self.screen.fill((0, 0, 0))
        
        # Draw paddles and ball
        pygame.draw.rect(self.screen, (255, 255, 255), self.left_paddle)
        pygame.draw.rect(self.screen, (255, 255, 255), self.right_paddle)
        pygame.draw.ellipse(self.screen, (255, 255, 255), self.ball)
        
        # Draw center line
        pygame.draw.aaline(self.screen, (255, 255, 255), (WIDTH // 2, 0), (WIDTH // 2, HEIGHT))
        
        # Draw scores
        score_text = f"{self.left_score} - {self.right_score}"
        text_surface = self.font.render(score_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(WIDTH // 2, 30))
        self.screen.blit(text_surface, text_rect)
        
        pygame.display.flip()

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.7,
    )
    mp_drawing = mp.solutions.drawing_utils

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot initiate camera")
        sys.exit()

    # Create game instance
    game = PongGame()
    running = True

    while running:
        # Compute delta time in seconds
        dt = game.clock.tick(60) / 1000.0

        # Process Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Initialize paddle movement deltas
        left_delta = 0
        right_delta = 0

        # Process hand tracking
        success, frame = cap.read()
        if not success:
            print("Failed to capture camera frame")
            break

        # Flip frame horizontally for a selfie view
        frame = cv2.flip(frame, 1)
        
        # Convert BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hands in the frame
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get hand position using mediapipe_cheats
                handedness = results.multi_handedness[idx].classification[0].label
                hand_x, hand_y = mpc.get_hand_center(hand_landmarks)
                
                # First ensure hand position is within control area
                hand_y = max(control_area["y_min"], min(control_area["y_max"], hand_y))
                
                # Calculate relative position within control area (0-1)
                relative_y = (hand_y - control_area["y_min"]) / (control_area["y_max"] - control_area["y_min"])
                
                # Map to game coordinates
                paddle_y = relative_y * HEIGHT
                
                if handedness == "Left":
                    # Set paddle position directly
                    target_y = paddle_y - PADDLE_HEIGHT / 2
                    current_y = game.left_paddle.y
                    left_delta = (target_y - current_y) * HAND_SENSITIVITY * 0.5  # Reduced sensitivity
                elif handedness == "Right":
                    # Set paddle position directly
                    target_y = paddle_y - PADDLE_HEIGHT / 2
                    current_y = game.right_paddle.y
                    right_delta = (target_y - current_y) * HAND_SENSITIVITY * 0.5  # Reduced sensitivity

        # Draw control area rectangle
        frame_height, frame_width = frame.shape[:2]
        x1 = int(control_area["x_min"] * frame_width)
        y1 = int(control_area["y_min"] * frame_height)
        x2 = int(control_area["x_max"] * frame_width)
        y2 = int(control_area["y_max"] * frame_height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Show the camera feed
        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key in OpenCV window
            break

        # Update game state and render
        game.move_paddles(left_delta, right_delta)
        game.update_ball(dt)
        game.draw()

    # Cleanup resources
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
