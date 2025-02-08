import mediapipe as mp
import numpy as np
import cv2

# Initialize MediaPipe hands solution for use in utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def process_frame(hands, frame):
    """
    Process a frame through MediaPipe hands detection
    
    Args:
        hands: MediaPipe Hands solution instance
        frame: BGR image/frame to process
    
    Returns:
        MediaPipe hand detection results
    """
    # Convert BGR to RGB since MediaPipe expects RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return hands.process(rgb_frame)

def draw_landmarks(frame, hand_landmarks):
    """
    Draw hand landmarks and connections on frame
    
    Args:
        frame: Image/frame to draw on
        hand_landmarks: MediaPipe hand landmarks for one hand
    """
    mp_drawing.draw_landmarks(
        frame, 
        hand_landmarks, 
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 100, 255), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
    )

def get_hand_type(handedness):
    """
    Get whether the hand is left or right
    
    Args:
        handedness: MediaPipe handedness classification result
    
    Returns:
        str: "Left" or "Right"
    """
    return handedness.classification[0].label

def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two landmarks
    
    Args:
        point1: First landmark point
        point2: Second landmark point
    
    Returns:
        float: Distance between points
    """
    return np.sqrt(
        (point1.x - point2.x)**2 + 
        (point1.y - point2.y)**2 + 
        (point1.z - point2.z)**2
    )

def detect_pinch(hand_landmarks, threshold=0.05):
    """
    Detect pinch gesture between thumb and index finger
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        threshold: Maximum distance to consider as a pinch
    
    Returns:
        bool: True if pinch detected, False otherwise
    """
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return calculate_distance(thumb_tip, index_tip) < threshold

def is_finger_raised(hand_landmarks, finger_name):
    """
    Check if a specific finger is raised
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        finger_name: String name of finger ('INDEX', 'MIDDLE', 'RING', 'PINKY')
    
    Returns:
        bool: True if finger is raised, False if lowered
    """
    # Define landmark mappings for each finger
    finger_landmarks = {
        'THUMB': (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_MCP),
        'INDEX': (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP),
        'MIDDLE': (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP),
        'RING': (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_MCP),
        'PINKY': (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_MCP)
    }
    
    if finger_name not in finger_landmarks:
        raise ValueError(f"Unknown finger name: {finger_name}")
        
    tip_landmark, base_landmark = finger_landmarks[finger_name]
    tip = hand_landmarks.landmark[tip_landmark]
    base = hand_landmarks.landmark[base_landmark]
    
    # For thumb, check x-position instead of y
    if finger_name == 'THUMB':
        return tip.x < base.x
    return tip.y < base.y

def get_hand_center(hand_landmarks):
    """
    Calculate the center point of the hand
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
    
    Returns:
        dict: Contains x, y coordinates of hand center
    """
    # Use multiple points to get a stable center
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_base = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    ring_base = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    
    # Average the positions
    x = (wrist.x + middle_base.x + index_base.x + ring_base.x) / 4
    y = (wrist.y + middle_base.y + index_base.y + ring_base.y) / 4
    
    return x, y
    #return {'x': x, 'y': y}

def calculate_hand_distance(left_landmarks, right_landmarks):
    """
    Calculate the distance between the centers of two hands using tuple coordinates
    
    Args:d
        left_landmarks: MediaPipe landmarks for left hand
        right_landmarks: MediaPipe landmarks for right hand
        
    Returns:
        float: Distance between hand centers
    """
    # Get the center points of each hand
    left_x, left_y = get_hand_center(left_landmarks)
    right_x, right_y = get_hand_center(right_landmarks)
    
    # Calculate Euclidean distance using the x,y coordinates
    return np.sqrt(
        (left_x - right_x)**2 + 
        (left_y - right_y)**2
    )

def get_hand_gesture(hand_landmarks):
    """
    Detect common hand gestures
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
    
    Returns:
        str: Name of detected gesture
    """
    # Check for pinch
    if detect_pinch(hand_landmarks):
        return "PINCH"
    
    # Count raised fingers
    raised_fingers = sum(1 for finger in ['INDEX', 'MIDDLE', 'RING', 'PINKY'] 
                        if is_finger_raised(hand_landmarks, finger))
    
    # Identify common gestures
    if raised_fingers == 0:
        return "FIST"
    elif raised_fingers <= 4:
        return "OPEN_PALM"
    elif raised_fingers == 1 and is_finger_raised(hand_landmarks, 'INDEX'):
        return "POINTING"
    elif raised_fingers == 2 and is_finger_raised(hand_landmarks, 'INDEX') and is_finger_raised(hand_landmarks, 'MIDDLE'):
        return "PEACE"
        
    return "UNKNOWN"

def detect_hand_direction(hand_landmarks):
    """
    Detect if a hand is pointing up, down, left, or right
    
    Args:
        hand_landmarks: MediaPipe hand landmarks

    Returns:
        str: Direction the hand is pointing ('UP', 'DOWN', 'LEFT', 'RIGHT')
        float: Angle of the hand in degrees
    """
    # Get the key points we'll use for direction detection
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    
    # Calculate the angle between the wrist and middle finger
    # We use the middle finger as it's usually the most stable for direction detection
    dx = middle_finger_tip.y - wrist.y
    dy = middle_finger_tip.x - wrist.x
    
    # Calculate angle in degrees
    angle = np.degrees(np.arctan2(dx, dy))
    
    # Normalize angle to 0-360 range
    angle = (angle + 360) % 360
    
    # Determine direction based on angle
    # We use 45-degree sectors for each direction
    if angle < 45 or angle >= 315:
        direction = "RIGHT"
    elif angle < 135:
        direction = "DOWN"
    elif angle < 225:
        direction = "LEFT"
    else:
        direction = "UP"
        
    return direction, angle

def get_hand_landmarks(results):

    """
    Organize hand landmarks by handedness (left/right)
    
    Returns:
        dict: Contains landmarks for each hand
            {
                'left': landmarks or None if no left hand,
                'right': landmarks or None if no right hand
            }
    """
    hands = {'left': None, 'right': None}
    
    # Only process if we detected any hands
    if results.multi_hand_landmarks:
        # Look at each detected hand
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get whether this is a left or right hand
            handedness = results.multi_handedness[idx].classification[0].label.lower()
            # Store the landmarks in the appropriate category
            hands[handedness] = hand_landmarks
    
    return hands

def detect_thumbs_up(hand_landmarks):
    """
    Detect if the hand is making a thumbs up gesture.
    
    A thumbs up gesture is characterized by:
    1. Thumb extended upward (lower y position than base)
    2. All other fingers closed (curled inward)
    3. Thumb should be significantly above other finger tips
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        
    Returns:
        bool: True if thumbs up detected, False otherwise
    """
    # Get thumb landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    
    # Get other finger tips
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # Get finger bases (MCP joints)
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    
    # Check if thumb is extended upward
    thumb_extended = thumb_tip.y < thumb_mcp.y
    
    # Check if thumb is significantly above other finger tips
    thumb_highest = all(
        thumb_tip.y < tip.y - 0.12  # Thumb should be significantly higher
        for tip in [index_tip, middle_tip, ring_tip, pinky_tip]
    )
    
    # Check if other fingers are closed by comparing tips to their bases
    fingers_closed = all(
        tip.y > base.y  # Tip should be below base for closed fingers
        for tip, base in [
            (index_tip, index_mcp),
            (middle_tip, middle_mcp),
            (ring_tip, ring_mcp),
            (pinky_tip, pinky_mcp)
        ]
    )
    
    return thumb_extended and thumb_highest and fingers_closed

def detect_thumbs_gesture(hand_landmarks, handedness, THRESHOLD = 30):
    """
    Detect thumbs up/down gesture using angle thresholds.
    
    The gesture is detected based on:
    1. Fingers being curled inward
    2. Thumb extension direction (left for right hand, right for left hand)
    3. Thumb angle relative to vertical (within 30 degrees of up/down)
    
    Args:
        hand_landmarks: MediaPipe hand landmarks
        handedness: String indicating "Left" or "Right" hand
        
    Returns:
        str: 'UP' for thumbs up, 'DOWN' for thumbs down, 'NONE' for no thumb gesture
    """
    # Get thumb landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
    
    # Get finger landmarks
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    index_base = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    middle_base = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    ring_base = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_base = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    base_list = [index_base, middle_base, ring_base, pinky_base]

    # Check if fingers are curled inward
    # fingers_curled = all(
    #     abs(tip.x - wrist.x) < abs(pip.x - wrist.x)
    #     for tip, pip in [
    #         (index_tip, index_pip),
    #         (middle_tip, middle_pip),
    #         (ring_tip, ring_pip),
    #         (pinky_tip, pinky_pip)
    #     ]
    # )

    if handedness == "Right":
        fingers_curled = all(
            tip.x > index_base.x + 0.02
            for tip, base in [
            (index_tip, index_base),
            (middle_tip, middle_base),
            (ring_tip, ring_base),
            (pinky_tip, pinky_base)   
            ]
        ) 

    else:
        fingers_curled = all(
            tip.x < index_base.x - 0.02 
            for tip, base in [
            (index_tip, index_base),
            (middle_tip, middle_base),
            (ring_tip, ring_base),
            (pinky_tip, pinky_base)   
            ]
        )
    
    correct_rotation = all(
        base.x < thumb_tip.x
        for base in base_list
    )

    if not fingers_curled:
        return "NONE"
    
    if not correct_rotation:
        return "NONE"

    # Calculate the angle between the thumb and the vertical axis
    # First, get the vector from MCP to tip
    dx = thumb_tip.x - thumb_mcp.x
    dy = thumb_tip.y - thumb_mcp.y
    
    # Calculate angle in degrees using arctangent
    # atan2 returns angle in radians in range (-π, π)
    angle = np.degrees(np.arctan2(dx, -dy))  # Negative dy because y increases downward
    
    # Normalize angle to be positive for both hands
    if handedness == "Left":
        angle = -angle
    
    # Determine gesture based on angle
    if abs(angle) <= THRESHOLD:
        return "UP"
    elif abs(angle) >= 180 - THRESHOLD:
        return "DOWN"
    else:
        return "NONE"