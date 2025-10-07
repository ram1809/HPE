import math

class MovementTracker:
    """
    Track the movement/displacement of a person in 3D space.
    """
    def __init__(self):
        # Initial position (set on first frame)
        self.start_x = None
        self.start_y = None
        self.start_z = None
        
        # Previous position (for incremental distance)
        self.prev_x = None
        self.prev_y = None
        self.prev_z = None
        
        # Current position
        self.current_x = None
        self.current_y = None
        self.current_z = None
        
        # Tracking stats
        self.total_distance = 0.0
        self.frame_count = 0
        self.unit_scale = 1.0  # Default scale factor (can be set to convert to mm)
        
    def set_unit_scale(self, scale):
        """Set scale factor to convert units (e.g., 1000 for meters to mm)"""
        self.unit_scale = scale
        
    def update_position(self, x, y, z):
        """
        Update the current position and calculate displacement and total distance.
        
        Args:
            x, y, z: Current coordinates
            
        Returns:
            tuple: (incremental_distance, displacement_from_start, total_distance)
        """
        # Store current position
        self.current_x = x
        self.current_y = y
        self.current_z = z
        self.frame_count += 1
        
        # If this is the first frame, initialize positions
        if self.start_x is None:
            self.start_x = x
            self.start_y = y
            self.start_z = z
            self.prev_x = x
            self.prev_y = y
            self.prev_z = z
            return 0.0, 0.0, 0.0
        
        # Calculate incremental distance from previous position
        incremental_distance = math.sqrt(
            (x - self.prev_x) ** 2 + 
            (y - self.prev_y) ** 2 + 
            (z - self.prev_z) ** 2
        ) * self.unit_scale
        
        # Calculate displacement (straight-line distance from start)
        displacement = math.sqrt(
            (x - self.start_x) ** 2 + 
            (y - self.start_y) ** 2 + 
            (z - self.start_z) ** 2
        ) * self.unit_scale
        
        # Update total distance
        self.total_distance += incremental_distance
        
        # Store current position as previous for next frame
        self.prev_x = x
        self.prev_y = y
        self.prev_z = z
        
        return incremental_distance, displacement, self.total_distance
    
    def get_stats(self):
        """Get movement statistics."""
        if self.frame_count == 0:
            return {
                'displacement': 0.0,
                'total_distance': 0.0,
                'frame_count': 0,
                'average_movement_per_frame': 0.0
            }
            
        # Calculate current displacement using current position
        if self.current_x is not None and self.start_x is not None:
            displacement = math.sqrt(
                (self.current_x - self.start_x) ** 2 + 
                (self.current_y - self.start_y) ** 2 + 
                (self.current_z - self.start_z) ** 2
            ) * self.unit_scale
        else:
            displacement = 0.0
            
        return {
            'displacement': displacement,
            'total_distance': self.total_distance,
            'frame_count': self.frame_count,
            'average_movement_per_frame': self.total_distance / self.frame_count
        }