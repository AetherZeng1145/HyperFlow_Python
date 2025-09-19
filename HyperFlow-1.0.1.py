import pygame
import math
import random
from colorsys import hsv_to_rgb
import time

# Initialize Pygame
pygame.init()
width, height = 1200, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("HyperFlow - Professional Wind Tunnel Simulator (Streamline Edition)")

# Use a system font to avoid display issues
try:
    font = pygame.font.SysFont('Arial', 16)
    font_large = pygame.font.SysFont('Arial', 20)
except pygame.error:
    # Fallback if Arial font is not available
    font = pygame.font.Font(None, 16)
    font_large = pygame.font.Font(None, 20)


# Define vector math functions
def vector_add(v1, v2):
    """Vector addition"""
    return [v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]]


def vector_sub(v1, v2):
    """Vector subtraction"""
    return [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]]


def vector_mul(v, scalar):
    """Vector scalar multiplication"""
    return [v[0] * scalar, v[1] * scalar, v[2] * scalar]


def vector_dot(v1, v2):
    """Vector dot product"""
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def vector_cross(v1, v2):
    """Vector cross product"""
    return [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    ]


def vector_norm(v):
    """Vector normalization"""
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length == 0:
        return [0, 0, 0]
    return [v[0] / length, v[1] / length, v[2] / length]


def vector_length(v):
    """Vector length"""
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def rotate_point(point, angles):
    """Rotate a point around X, Y, and Z axes"""
    x, y, z = point

    # Rotate around X-axis
    y_rot = y * math.cos(math.radians(angles[0])) - z * math.sin(math.radians(angles[0]))
    z_rot = y * math.sin(math.radians(angles[0])) + z * math.cos(math.radians(angles[0]))
    y, z = y_rot, z_rot

    # Rotate around Y-axis
    x_rot = x * math.cos(math.radians(angles[1])) + z * math.sin(math.radians(angles[1]))
    z_rot = -x * math.sin(math.radians(angles[1])) + z * math.cos(math.radians(angles[1]))
    x, z = x_rot, z_rot

    # Rotate around Z-axis
    x_rot = x * math.cos(math.radians(angles[2])) - y * math.sin(math.radians(angles[2]))
    y_rot = x * math.sin(math.radians(angles[2])) + y * math.cos(math.radians(angles[2]))
    x, y = x_rot, y_rot

    return [x, y, z]


# Model definition
class Model:
    def __init__(self, name, vertices, edges, faces, color, characteristics):
        self.name = name
        self.vertices = vertices
        self.edges = edges
        self.faces = faces
        self.base_color = color
        self.angles = [0, 0, 0]  # X, Y, Z rotation angles
        self.position = [0, 0, 0]
        self.pressure_map = {}  # Stores pressure values for each face
        self.characteristics = characteristics  # Model characteristics
        self.control_surfaces = {
            "flaps": 0,  # Flap angle (degrees)
            "spoilers": 0,  # Spoiler deployment (0-100%)
            "elevator": 0,  # Elevator angle (degrees)
            "rudder": 0  # Rudder angle (degrees)
        }
        self.face_normals = []
        self.calculate_normals()

    def calculate_normals(self):
        """Calculates the normal vector for each face, with robust index checking."""
        self.face_normals = []
        max_vertex_index = len(self.vertices) - 1

        for i, face in enumerate(self.faces):
            # Check for invalid face definitions
            if len(face) < 3 or any(idx < 0 or idx > max_vertex_index for idx in face):
                self.face_normals.append([0.0, 0.0, 0.0])  # Add a placeholder normal
                continue

            # Use a try-except block for added safety
            try:
                # Get the rotated vertices
                rotated_vertices = [rotate_point(v, self.angles) for v in self.vertices]
                v1 = vector_sub(rotated_vertices[face[1]], rotated_vertices[face[0]])
                v2 = vector_sub(rotated_vertices[face[2]], rotated_vertices[face[0]])
                normal = vector_cross(v1, v2)
                normal = vector_norm(normal)  # Normalize
                self.face_normals.append(normal)
            except IndexError:
                self.face_normals.append([0.0, 0.0, 0.0])

    def update_pressure(self, wind_direction, wind_speed):
        """Updates the pressure distribution based on wind direction and speed"""
        self.pressure_map = {}
        # Recalculate normals to account for model angle
        self.calculate_normals()
        for i, normal in enumerate(self.face_normals):
            # Calculate the dot product between the wind and the normal vector
            angle = vector_dot(normal, wind_direction)
            # Pressure is related to wind speed and angle
            pressure = wind_speed * max(0, angle) * self.characteristics["pressure_factor"]
            self.pressure_map[i] = pressure

    def draw(self, screen, camera_angle, wireframe_only=False):
        """
        Draws the 3D model, supporting wireframe and pressure-colored modes
        """
        # Simple 3D to 2D projection
        projected_points = []
        for vertex in self.vertices:
            # Apply model rotation
            rotated_vertex = rotate_point(vertex, self.angles)

            # Apply control surface adjustments
            if hasattr(self, 'apply_control_surfaces'):
                rotated_vertex = self.apply_control_surfaces(rotated_vertex)

            # Translation
            x = rotated_vertex[0] + self.position[0]
            y = rotated_vertex[1] + self.position[1]
            z = rotated_vertex[2] + self.position[2]

            # Apply camera rotation
            camera_rotated = rotate_point([x, y, z], [-camera_angle[0], -camera_angle[1], 0])
            x, y, z = camera_rotated

            # Perspective projection, with an epsilon to prevent division by zero
            epsilon = 0.001
            factor = 200 / (z + 5 + epsilon)
            px = width / 2 + x * factor
            py = height / 2 - y * factor

            projected_points.append((px, py, z))

        # Sort faces by depth (simple painter's algorithm)
        sorted_faces = []
        for i, face in enumerate(self.faces):
            # Calculate average depth of the face
            avg_z = sum(projected_points[vertex][2] for vertex in face) / len(face)
            sorted_faces.append((i, avg_z))

        # Sort from back to front
        sorted_faces.sort(key=lambda x: x[1], reverse=True)

        # Draw faces unless in wireframe mode
        if not wireframe_only:
            for face_idx, _ in sorted_faces:
                face = self.faces[face_idx]

                # Adjust color based on pressure
                pressure = self.pressure_map.get(face_idx, 0)
                # Pressure color transitions from blue to red
                pressure_ratio = min(1.0, pressure / 1000.0)
                r = int(255 * pressure_ratio)
                b = int(255 * (1 - pressure_ratio))
                g = int(255 * 0.2)
                face_color = (r, g, b)

                # Draw the polygon
                points = [projected_points[vertex][:2] for vertex in face]
                if len(points) >= 3:
                    pygame.draw.polygon(screen, face_color, points)

        # Draw edges
        for edge in self.edges:
            start = projected_points[edge[0]][:2]
            end = projected_points[edge[1]][:2]
            pygame.draw.line(screen, (50, 50, 50), start, end, 2)

    def calculate_forces(self, wind_direction, wind_speed):
        """Calculates lift and drag based on real physics principles"""
        lift_force = 0
        drag_force = 0

        # Calculate Angle of Attack (AoA)
        aoa = math.degrees(math.acos(max(-1, min(1, vector_dot([0, 0, 1], wind_direction)))))
        if wind_direction[2] < 0:
            aoa = -aoa

        # Apply control surface effects
        cl_modifier = 1.0 + (self.control_surfaces["flaps"] / 30.0)  # Flaps increase lift
        cd_modifier = 1.0 + (self.control_surfaces["spoilers"] / 50.0)  # Spoilers increase drag
        aoa += self.control_surfaces["elevator"] / 2.0  # Elevator affects AoA

        # Calculate lift and drag coefficients based on model characteristics and AoA
        cl = (self.characteristics["cl_base"] + self.characteristics["cl_alpha"] * aoa) * cl_modifier
        cd = (self.characteristics["cd_base"] + self.characteristics["cd_alpha"] * aoa ** 2) * cd_modifier

        # Calculate force (F = 0.5 * ρ * v² * A * C)
        dynamic_pressure = 0.5 * 1.225 * (wind_speed * 50) ** 2
        lift_force = dynamic_pressure * cl
        drag_force = dynamic_pressure * cd

        return lift_force, drag_force, aoa


# Define detailed model data
def create_aircraft():
    # Modern Airliner Model (more realistic)
    vertices = [
        # Fuselage (more streamlined)
        [-1.8, -0.15, -0.15], [1.8, -0.15, -0.15], [1.8, 0.15, -0.15], [-1.8, 0.15, -0.15],
        [-1.8, -0.15, 0.15], [1.8, -0.15, 0.15], [1.8, 0.15, 0.15], [-1.8, 0.15, 0.15],
        # Nose
        [2.2, 0.0, 0.0],  # 8
        # Main wings (with sweep and dihedral)
        [0.0, -2.5, 0.0], [0.8, -2.5, 0.2], [0.8, 2.5, 0.2], [0.0, 2.5, 0.0],  # 9-12
        # Horizontal stabilizer
        [-1.5, -1.0, 0.1], [-1.0, -1.0, 0.2], [-1.0, 1.0, 0.2], [-1.5, 1.0, 0.1],  # 13-16
        # Vertical stabilizer
        [-1.5, 0.0, 0.1], [-1.0, 0.0, 0.2], [-1.0, 0.0, 1.0], [-1.5, 0.0, 0.9],  # 17-20
        # Engine nacelles
        [0.4, -1.8, 0.0], [0.7, -1.8, 0.0], [0.7, -1.8, 0.3], [0.4, -1.8, 0.3],  # 21-24
        [0.4, 1.8, 0.0], [0.7, 1.8, 0.0], [0.7, 1.8, 0.3], [0.4, 1.8, 0.3],  # 25-28
        # Winglets
        [1.0, -2.7, 0.3], [1.0, 2.7, 0.3],  # 29-30
    ]
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],  # Fuselage box
        [0, 4], [1, 5], [2, 6], [3, 7],
        [1, 8], [2, 8], [5, 8], [6, 8],  # Nose
        [9, 10], [10, 11], [11, 12], [12, 9],  # Main wings
        [13, 14], [14, 15], [15, 16], [16, 13],  # Horizontal stabilizer
        [17, 18], [18, 19], [19, 20], [20, 17],  # Vertical stabilizer
        [21, 22], [22, 23], [23, 24], [24, 21],  # Engine nacelle 1
        [25, 26], [26, 27], [27, 28], [28, 25],  # Engine nacelle 2
        [10, 29], [11, 30]  # Winglets
    ]
    faces = [
        # Fuselage faces (corrected)
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 4, 7, 3], [1, 5, 6, 2], [0, 1, 5, 4], [2, 3, 7, 6],
        # Main wings
        [9, 10, 11], [12, 9, 11],
        # Horizontal stabilizer
        [13, 14, 15], [16, 13, 15],
        # Vertical stabilizer
        [17, 18, 19], [20, 17, 19],
        # Engine nacelles
        [21, 22, 23, 24], [25, 26, 27, 28]
    ]
    characteristics = {
        "cl_base": 0.3, "cl_alpha": 0.1, "cd_base": 0.05,
        "cd_alpha": 0.005, "pressure_factor": 1.0
    }
    model = Model("Modern Airliner", vertices, edges, faces, (0.8, 0.2, 0.2), characteristics)

    def apply_control_surfaces(self, vertex):
        x, y, z = vertex
        if y < -1.5 or y > 1.5:
            z += math.sin(math.radians(self.control_surfaces["flaps"])) * 0.1
        if x < -1.0 and abs(y) < 1.0:
            z += math.sin(math.radians(self.control_surfaces["elevator"])) * 0.1
        if x < -1.0 and abs(y) < 0.2:
            y += math.sin(math.radians(self.control_surfaces["rudder"])) * 0.1
        return [x, y, z]

    model.apply_control_surfaces = apply_control_surfaces.__get__(model, Model)
    model.calculate_normals()
    return model


def create_car():
    # Modern Sports Car Model (more realistic)
    vertices = [
        [-1.5, -0.4, -0.1], [1.5, -0.4, -0.1], [1.5, 0.4, -0.1], [-1.5, 0.4, -0.1],
        [-1.5, -0.4, 0.4], [1.5, -0.4, 0.4], [1.5, 0.4, 0.4], [-1.5, 0.4, 0.4],
        [1.5, -0.3, 0.1], [1.8, -0.3, 0.1], [1.8, 0.3, 0.1], [1.5, 0.3, 0.1],
        [1.8, -0.2, 0.1], [1.9, -0.2, 0.1], [1.9, 0.2, 0.1], [1.8, 0.2, 0.1],
        [-1.5, -0.5, 0.1], [-1.8, -0.5, 0.1], [-1.8, 0.5, 0.1], [-1.5, 0.5, 0.1],
        [-1.8, -0.4, 0.3], [-1.9, -0.4, 0.3], [-1.9, 0.4, 0.3], [-1.8, 0.4, 0.3],
        [-0.8, -0.5, -0.1], [-0.4, -0.5, -0.1], [-0.4, -0.5, 0.3], [-0.8, -0.5, 0.3],
        [0.8, -0.5, -0.1], [1.2, -0.5, -0.1], [1.2, -0.5, 0.3], [0.8, -0.5, 0.3],
        [-0.8, 0.5, -0.1], [-0.4, 0.5, -0.1], [-0.4, 0.5, 0.3], [-0.8, 0.5, 0.3],
        [0.8, 0.5, -0.1], [1.2, 0.5, -0.1], [1.2, 0.5, 0.3], [0.8, 0.5, 0.3],
        [0.5, -0.4, 0.4], [1.0, -0.4, 0.6], [1.0, 0.4, 0.6], [0.5, 0.4, 0.4],
        [1.3, -0.6, 0.3], [1.4, -0.6, 0.3], [1.4, -0.6, 0.4], [1.3, -0.6, 0.4],
        [1.3, 0.6, 0.3], [1.4, 0.6, 0.3], [1.4, 0.6, 0.4], [1.3, 0.6, 0.4],
    ]
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7], [1, 8], [8, 9], [9, 10], [10, 11],
        [11, 2], [5, 8], [6, 11], [9, 12], [12, 13], [13, 14], [14, 10],
        [0, 15], [15, 16], [16, 17], [17, 18], [18, 3], [4, 15], [7, 18],
        [16, 19], [19, 20], [20, 21], [21, 17], [22, 23], [23, 24], [24, 25],
        [25, 22], [26, 27], [27, 28], [28, 29], [29, 26], [30, 31], [31, 32],
        [32, 33], [33, 30], [34, 35], [35, 36], [36, 37], [37, 34], [4, 38],
        [5, 39], [38, 39], [39, 40], [40, 41], [41, 7], [6, 40], [7, 41],
        [42, 43], [43, 44], [44, 45], [45, 42], [46, 47], [47, 48], [48, 49],
        [49, 46],
    ]
    # Corrected face list to match the provided vertices
    faces = [
        # Main body (box)
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5],
        [2, 3, 7, 6], [3, 0, 4, 7],
        # Hood and back window
        [1, 8, 11, 2], [5, 8, 11, 6],
        # Headlights
        [9, 10, 11, 8], [12, 13, 14, 15],
        # Trunk
        [0, 15, 16, 17, 18, 3], [4, 15, 16, 17, 18, 7],
        # Taillights
        [16, 19, 20, 21, 17], [22, 23, 24, 25],
        # Wheel arches
        [26, 27, 28, 29], [30, 31, 32, 33],
        [34, 35, 36, 37], [38, 39, 40, 41],
        # Spoiler
        [42, 43, 44, 45], [46, 47, 48, 49]
    ]

    characteristics = {
        "cl_base": 0.1, "cl_alpha": 0.02, "cd_base": 0.3,
        "cd_alpha": 0.01, "pressure_factor": 0.8
    }
    model = Model("Sports Car", vertices, edges, faces, (0.2, 0.6, 0.8), characteristics)
    model.calculate_normals()
    return model


# Create list of models
models = {
    "Modern Airliner": create_aircraft(),
    "Sports Car": create_car()
}

current_model = models["Modern Airliner"]


# Streamline particles
class FlowParticle:
    def __init__(self, pos, max_trail_length):
        self.pos = pos
        self.trail = []  # Stores the position trail to draw the streamline
        self.max_trail_length = max_trail_length  # Trail length

    def update(self, new_pos, max_trail_length):
        self.pos = new_pos
        self.max_trail_length = max_trail_length
        self.trail.append(new_pos)
        # Limit the trail length
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)


# Create wind tunnel environment
class WindTunnel:
    def __init__(self):
        self.wind_speed = 10.0  # Wind speed strength (m/s)
        self.wind_direction = [1.0, 0.0, 0.0]  # Wind direction
        self.wind_direction = vector_norm(self.wind_direction)  # Normalize
        self.turbulence = 0.1  # Turbulence strength
        self.particles = []  # List of particles
        self.max_particles = 1000  # Max number of particles
        self.spawn_rate = 10  # Number of particles to spawn per frame
        self.lift_force = 0
        self.drag_force = 0
        self.aoa = 0  # Angle of Attack
        self.trail_length = 50

    def update_particles(self, model):
        # Add new particles
        if len(self.particles) < self.max_particles:
            for _ in range(self.spawn_rate):
                y_offset = random.uniform(-2, 2)
                z_offset = random.uniform(-1, 1)
                new_particle = FlowParticle([-5.0, y_offset, z_offset], self.trail_length)
                self.particles.append(new_particle)

        # Update existing particles
        model_center = [0, 0, 0]
        for particle in self.particles:
            # Initial wind velocity
            base_velocity = vector_mul(self.wind_direction, self.wind_speed * 0.02)

            # Simplified repulsion logic
            dist_to_center = vector_length(vector_sub(particle.pos, model_center))
            repulsion_force = [0, 0, 0]

            # Apply repulsion force only when particle is near the model
            if dist_to_center < 1.5:
                deflection_strength = 1.0 / (dist_to_center + 0.01)
                direction = vector_norm(vector_sub(particle.pos, model_center))
                repulsion_force = vector_mul(direction, deflection_strength * 0.05)

            # Add turbulence
            turbulence_force = [
                random.uniform(-self.turbulence, self.turbulence) * 0.05,
                random.uniform(-self.turbulence, self.turbulence) * 0.05,
                random.uniform(-self.turbulence, self.turbulence) * 0.05
            ]

            # Update position
            new_pos = vector_add(particle.pos, base_velocity)
            new_pos = vector_add(new_pos, repulsion_force)
            new_pos = vector_add(new_pos, turbulence_force)
            particle.update(new_pos, self.trail_length)

        # Remove old particles that have left the screen
        self.particles = [p for p in self.particles if p.pos[0] < 5]

        # Update model pressure and forces
        model.update_pressure(self.wind_direction, self.wind_speed)
        self.lift_force, self.drag_force, self.aoa = model.calculate_forces(self.wind_direction, self.wind_speed)

    def draw_particles(self, screen, camera_angle):
        for particle in self.particles:
            if len(particle.trail) < 2:
                continue

            # Project trail points to the screen
            projected_trail = []
            for pos in particle.trail:
                camera_rotated = rotate_point(pos, [-camera_angle[0], -camera_angle[1], 0])
                x, y, z = camera_rotated
                epsilon = 0.001
                factor = 200 / (z + 5 + epsilon)
                px = width / 2 + x * factor
                py = height / 2 - y * factor
                projected_trail.append((px, py))

            # Draw the streamline
            if len(projected_trail) >= 2:
                # Draw the streamline and make the tail fade out
                for i in range(len(projected_trail) - 1):
                    # Calculate color based on position to create a gradient
                    # Fix for ValueError due to float precision
                    # age_ratio now changes smoothly from 0 (tail) to 1 (head)
                    age_ratio = (len(projected_trail) - 1 - i) / (len(projected_trail) - 1)

                    # Use interpolation to create a fade from background to bright color for a smooth fade-out effect at the tail
                    bg_color = (20, 20, 30)
                    line_color = (150, 200, 255)
                    r = int(bg_color[0] + (line_color[0] - bg_color[0]) * (1 - age_ratio))
                    g = int(bg_color[1] + (line_color[1] - bg_color[1]) * (1 - age_ratio))
                    b = int(bg_color[2] + (line_color[2] - bg_color[2]) * (1 - age_ratio))
                    color = (r, g, b)

                    pygame.draw.line(screen, color, projected_trail[i], projected_trail[i + 1], 1)


# Create wind tunnel instance
wind_tunnel = WindTunnel()


# Create UI controls
class Slider:
    def __init__(self, x, y, width, height, min_val, max_val, initial, label):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial
        self.label = label
        self.dragging = False

    def draw(self, surface):
        pygame.draw.rect(surface, (100, 100, 100), (self.x, self.y, self.width, self.height))
        pos_x = self.x + int((self.value - self.min_val) / (self.max_val - self.min_val) * self.width)
        pygame.draw.rect(surface, (200, 200, 200), (pos_x - 5, self.y - 5, 10, self.height + 10))
        label_text = font.render(f"{self.label}: {self.value:.1f}", True, (255, 255, 255))
        surface.blit(label_text, (self.x, self.y - 20))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = pygame.mouse.get_pos()
            if (self.x <= mouse_pos[0] <= self.x + self.width and
                    self.y - 5 <= mouse_pos[1] <= self.y + self.height + 5):
                self.dragging = True
                self.update_value(mouse_pos[0])
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self.update_value(event.pos[0])

    def update_value(self, x_pos):
        x_pos = max(self.x, min(self.x + self.width, x_pos))
        self.value = self.min_val + (x_pos - self.x) / self.width * (self.max_val - self.min_val)


class Dropdown:
    def __init__(self, x, y, width, height, options):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.options = options
        self.selected = options[0]
        self.active = False
        self.rect = pygame.Rect(x, y, width, height)

    def draw(self, surface):
        color = (100, 100, 100) if not self.active else (150, 150, 150)
        pygame.draw.rect(surface, color, self.rect)
        text = font.render(self.selected, True, (255, 255, 255))
        surface.blit(text, (self.x + 5, self.y + 5))
        pygame.draw.polygon(surface, (255, 255, 255), [
            (self.x + self.width - 20, self.y + 10),
            (self.x + self.width - 10, self.y + 10),
            (self.x + self.width - 15, self.y + 20)
        ])
        if self.active:
            for i, option in enumerate(self.options):
                y_pos = self.y + self.height + i * self.height
                pygame.draw.rect(surface, (100, 100, 100), (self.x, y_pos, self.width, self.height))
                text = font.render(option, True, (255, 255, 255))
                surface.blit(text, (self.x + 5, y_pos + 5))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = event.pos
            if self.rect.collidepoint(mouse_pos):
                self.active = not self.active
            elif self.active:
                for i, option in enumerate(self.options):
                    y_pos = self.y + self.height + i * self.height
                    option_rect = pygame.Rect(self.x, y_pos, self.width, self.height)
                    if option_rect.collidepoint(mouse_pos):
                        self.selected = option
                        self.active = False
                        return True
            else:
                self.active = False
        return False


# Create UI elements
wind_slider = Slider(50, 50, 200, 20, 0, 30, 10, "Wind Speed (m/s)")
angle_slider = Slider(50, 100, 200, 20, -30, 30, 0, "Attitude (degrees)")
particle_count_slider = Slider(50, 150, 200, 20, 100, 5000, 1000, "Particle Count")
trail_length_slider = Slider(50, 200, 200, 20, 10, 200, 50, "Trail Length")
model_dropdown = Dropdown(50, 250, 200, 30, list(models.keys()))

# Aircraft control surface sliders
flaps_slider = Slider(50, 350, 200, 20, 0, 30, 0, "Flaps (degrees)")
spoilers_slider = Slider(50, 400, 200, 20, 0, 100, 0, "Spoilers (%)")
elevator_slider = Slider(50, 450, 200, 20, -20, 20, 0, "Elevator (degrees)")
rudder_slider = Slider(50, 500, 200, 20, -20, 20, 0, "Rudder (degrees)")


# Pressure legend
def draw_pressure_legend(surface):
    title_text = font_large.render("Air Resistance / Pressure", True, (255, 255, 255))
    surface.blit(title_text, (width - 200, 30))
    s = pygame.Surface((120, 200), pygame.SRCALPHA)
    s.fill((50, 50, 50, 180))
    surface.blit(s, (width - 150, 60))
    for i in range(100):
        pressure_ratio = i / 100.0
        r = int(255 * pressure_ratio)
        b = int(255 * (1 - pressure_ratio))
        g = int(255 * 0.2)
        color = (r, g, b)
        pygame.draw.rect(surface, color,
                         (width - 140, 60 + i * 2, 100, 2))
    low_text = font.render("Low", True, (255, 255, 255))
    high_text = font.render("High", True, (255, 255, 255))
    surface.blit(low_text, (width - 140, 265))
    surface.blit(high_text, (width - 140, 50))


# Data display panel
def draw_data_panel(surface, lift, drag, aoa, model_name, wind_speed):
    s = pygame.Surface((280, 160), pygame.SRCALPHA)
    s.fill((40, 40, 40, 180))
    surface.blit(s, (width - 300, height - 180))

    title_text = font_large.render("Wind Tunnel Data", True, (255, 255, 255))
    surface.blit(title_text, (width - 290, height - 170))
    lift_text = font.render(f"Lift: {lift:.1f} N", True, (0, 255, 0))
    drag_text = font.render(f"Drag: {drag:.1f} N", True, (255, 0, 0))
    aoa_text = font.render(f"AoA: {aoa:.1f}°", True, (200, 200, 100))
    model_text = font.render(f"Model: {model_name}", True, (200, 200, 200))
    wind_text = font.render(f"Wind Speed: {wind_speed:.1f} m/s", True, (100, 200, 255))
    surface.blit(lift_text, (width - 290, height - 140))
    surface.blit(drag_text, (width - 290, height - 120))
    surface.blit(aoa_text, (width - 290, height - 100))
    surface.blit(model_text, (width - 290, height - 80))
    surface.blit(wind_text, (width - 290, height - 60))
    if drag > 0:
        l_d_ratio = lift / drag
        ratio_text = font.render(f"Lift/Drag Ratio: {l_d_ratio:.2f}", True, (255, 255, 0))
        surface.blit(ratio_text, (width - 150, height - 140))


# Main loop
clock = pygame.time.Clock()
running = True
last_time = time.time()
camera_angle = [0, 0]
mouse_down = False
last_mouse_pos = (0, 0)
zoom_speed = 0.5
wireframe_mode = False
wireframe_toggle_time = 0

while running:
    current_time = time.time()
    last_time = current_time

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_down = True
                last_mouse_pos = pygame.mouse.get_pos()
            elif event.button == 4:
                # Mouse wheel up zooms in
                pass
            elif event.button == 5:
                # Mouse wheel down zooms out
                pass
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            mouse_down = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                wireframe_mode = not wireframe_mode
                wireframe_toggle_time = current_time
        wind_slider.handle_event(event)
        angle_slider.handle_event(event)
        particle_count_slider.handle_event(event)
        trail_length_slider.handle_event(event)

        if model_dropdown.handle_event(event):
            current_model = models[model_dropdown.selected]
            wind_tunnel.particles = []
        if current_model.name == "Modern Airliner":
            flaps_slider.handle_event(event)
            spoilers_slider.handle_event(event)
            elevator_slider.handle_event(event)
            rudder_slider.handle_event(event)
            current_model.control_surfaces["flaps"] = flaps_slider.value
            current_model.control_surfaces["spoilers"] = spoilers_slider.value
            current_model.control_surfaces["elevator"] = elevator_slider.value
            current_model.control_surfaces["rudder"] = rudder_slider.value

    if mouse_down:
        current_mouse_pos = pygame.mouse.get_pos()
        dx = current_mouse_pos[0] - last_mouse_pos[0]
        dy = current_mouse_pos[1] - last_mouse_pos[1]
        camera_angle[1] += dx * 0.5
        camera_angle[0] += dy * 0.5
        camera_angle[0] = max(-90, min(90, camera_angle[0]))
        last_mouse_pos = current_mouse_pos

    current_model.angles[0] = angle_slider.value
    wind_tunnel.wind_speed = wind_slider.value
    wind_tunnel.max_particles = int(particle_count_slider.value)
    wind_tunnel.trail_length = int(trail_length_slider.value)
    wind_tunnel.update_particles(current_model)
    screen.fill((20, 20, 30))

    # Draw background grid
    for i in range(-10, 11):
        start_x, end_x = -5, 5
        y = i * 0.5
        start_rotated = rotate_point([start_x, y, 0], [-camera_angle[0], -camera_angle[1], 0])
        end_rotated = rotate_point([end_x, y, 0], [-camera_angle[0], -camera_angle[1], 0])
        epsilon = 0.001
        factor1 = 200 / (start_rotated[2] + 5 + epsilon)
        factor2 = 200 / (end_rotated[2] + 5 + epsilon)
        px1 = width / 2 + start_rotated[0] * factor1
        py1 = height / 2 - start_rotated[1] * factor1
        px2 = width / 2 + end_rotated[0] * factor2
        py2 = height / 2 - end_rotated[1] * factor2
        pygame.draw.line(screen, (50, 50, 70), (px1, py1), (px2, py2), 1)

        x = i * 0.5
        start_rotated = rotate_point([x, -5, 0], [-camera_angle[0], -camera_angle[1], 0])
        end_rotated = rotate_point([x, 5, 0], [-camera_angle[0], -camera_angle[1], 0])
        factor1 = 200 / (start_rotated[2] + 5 + epsilon)
        factor2 = 200 / (end_rotated[2] + 5 + epsilon)
        px1 = width / 2 + start_rotated[0] * factor1
        py1 = height / 2 - start_rotated[1] * factor1
        px2 = width / 2 + end_rotated[0] * factor2
        py2 = height / 2 - end_rotated[1] * factor2
        pygame.draw.line(screen, (50, 50, 70), (px1, py1), (px2, py2), 1)

    wind_tunnel.draw_particles(screen, camera_angle)
    current_model.draw(screen, camera_angle, wireframe_mode)
    wind_slider.draw(screen)
    angle_slider.draw(screen)
    particle_count_slider.draw(screen)
    trail_length_slider.draw(screen)
    model_dropdown.draw(screen)

    if current_model.name == "Modern Airliner":
        flaps_slider.draw(screen)
        spoilers_slider.draw(screen)
        elevator_slider.draw(screen)
        rudder_slider.draw(screen)

    draw_pressure_legend(screen)
    draw_data_panel(screen, wind_tunnel.lift_force, wind_tunnel.drag_force, wind_tunnel.aoa,
                    current_model.name, wind_tunnel.wind_speed)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
