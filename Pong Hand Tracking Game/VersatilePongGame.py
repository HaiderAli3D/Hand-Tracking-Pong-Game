import pygame
import sys

# Game constants
WIDTH, HEIGHT = 800, 600
PADDLE_WIDTH, PADDLE_HEIGHT = 6, 120
BALL_SIZE = 10
PADDLE_SPEED = 5
BALL_SPEED_X, BALL_SPEED_Y = 3, 3

class PongGame:
    def __init__(self):
        # Initialize Pygame and create the game window.
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("2 Player Pong Game")
        self.clock = pygame.time.Clock()

        # Define the left and right paddles.
        self.left_paddle = pygame.Rect(10, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.right_paddle = pygame.Rect(WIDTH - 10 - PADDLE_WIDTH, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
        
        # Define the ball.
        self.ball = pygame.Rect(WIDTH // 2 - BALL_SIZE // 2, HEIGHT // 2 - BALL_SIZE // 2, BALL_SIZE, BALL_SIZE)
        self.ball_speed_x = BALL_SPEED_X
        self.ball_speed_y = BALL_SPEED_Y

        # Initialize scores for both players.
        self.left_score = 0
        self.right_score = 0

        # Set up the font for the score display.
        self.font = pygame.font.Font(None, 36)

    def move_paddles(self, left_delta, right_delta):
        """
        Move the left and right paddles by the given deltas while ensuring they remain on screen.
        :param left_delta: Change in position for the left paddle.
        :param right_delta: Change in position for the right paddle.
        """
        # Update left paddle position.
        self.left_paddle.y += left_delta
        if self.left_paddle.top < 0:
            self.left_paddle.top = 0
        if self.left_paddle.bottom > HEIGHT:
            self.left_paddle.bottom = HEIGHT

        # Update right paddle position.
        self.right_paddle.y += right_delta
        if self.right_paddle.top < 0:
            self.right_paddle.top = 0
        if self.right_paddle.bottom > HEIGHT:
            self.right_paddle.bottom = HEIGHT

    def update_ball(self):
        """Update the ball's position, handle collisions, and update scores."""
        # Move the ball.
        self.ball.x += self.ball_speed_x
        self.ball.y += self.ball_speed_y

        # Bounce off the top and bottom walls.
        if self.ball.top <= 0 or self.ball.bottom >= HEIGHT:
            self.ball_speed_y *= -1

        # Bounce off the paddles.
        if self.ball.colliderect(self.left_paddle) or self.ball.colliderect(self.right_paddle):
            self.ball_speed_x *= -1

        # Update score if the ball leaves the screen.
        if self.ball.left <= 0:
            # Right player scores.
            self.right_score += 1
            self.reset_ball()
        elif self.ball.right >= WIDTH:
            # Left player scores.
            self.left_score += 1
            self.reset_ball()

    def reset_ball(self):
        """Reset the ball to the center and reverse its horizontal direction."""
        self.ball.center = (WIDTH // 2, HEIGHT // 2)
        self.ball_speed_x *= -1

    def draw(self):
        """Render all game elements including the paddles, ball, and score."""
        self.screen.fill((0, 0, 0))  # Clear the screen.
        pygame.draw.rect(self.screen, (255, 255, 255), self.left_paddle)
        pygame.draw.rect(self.screen, (255, 255, 255), self.right_paddle)
        pygame.draw.ellipse(self.screen, (255, 255, 255), self.ball)
        pygame.draw.aaline(self.screen, (255, 255, 255), (WIDTH // 2, 0), (WIDTH // 2, HEIGHT))
        
        # Render the score.
        score_text = f"{self.left_score} - {self.right_score}"
        text_surface = self.font.render(score_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(WIDTH // 2, 30))
        self.screen.blit(text_surface, text_rect)
        
        pygame.display.flip()

if __name__ == '__main__':
    # Create an instance of the game.
    game = PongGame()
    running = True

    # Main game loop.
    while running:
        # Cap the frame rate to 60 FPS.
        game.clock.tick(60)
        
        # Process events.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Exit if ESC is pressed.
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Detect continuous key presses.
        keys = pygame.key.get_pressed()
        left_delta = 0
        right_delta = 0

        # Left paddle movement using W and S.
        if keys[pygame.K_w]:
            left_delta = -PADDLE_SPEED
        elif keys[pygame.K_s]:
            left_delta = PADDLE_SPEED

        # Right paddle movement using Up and Down arrow keys.
        if keys[pygame.K_UP]:
            right_delta = -PADDLE_SPEED
        elif keys[pygame.K_DOWN]:
            right_delta = PADDLE_SPEED

        # Update paddle positions based on external input.
        game.move_paddles(left_delta, right_delta)
        
        # Update the ball and draw the current game state.
        game.update_ball()
        game.draw()

    pygame.quit()
    sys.exit()
