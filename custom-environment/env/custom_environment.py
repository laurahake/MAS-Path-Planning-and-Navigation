import numpy as np
import pygame
import math
import string
from node_class import Node
from gymnasium.utils import EzPickle
from gymnasium.core import ObsType
from typing import Any
from simple_pid import PID

from pettingzoo.mpe._mpe_utils.core import World as BaseWorld
from pettingzoo.mpe._mpe_utils.core import Landmark as BaseLandmark
from pettingzoo.mpe._mpe_utils.core import Agent as BaseAgent
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn

alphabet = list(string.ascii_uppercase)
        
def print_state(state, size=9):
    print("The state is:")
    for c in range(size-1, -1, -1):
        column = state[c::size] # every sizeth element
        print(" ".join(f"{cell:2}" for cell in column))
        
        
class Agent(BaseAgent):
    """
    Description: Agent object with the addition of a*-path attributes and the Q-Learning state
        
    """
    def __init__(self, Kp = 10, Ki = 0, Kd = 0):
        super().__init__()
        self.a_star_old = []
        self.a_star_new = []
        self.goal_point = []
        size = 9
        self.q_state = [1] * (size * size)
        self.controller_x = PID(Kp, Ki, Kd, setpoint=0)
        self.controller_y = PID(Kp, Ki, Kd, setpoint=0)
        self.terminated =  False


class Landmark(BaseLandmark):
    """
    Description: Used for the Agent goal points
        
    """ 
    def __init__(self):
        super().__init__()

    def is_collision(self, agent):
        euclidian_dis = math.sqrt((self.state.p_pos[0] - agent.state.p_pos[0]) ** 2 + (agent.state.p_pos[1] - agent.state.p_pos[1]) ** 2)
        if euclidian_dis <= agent.size:
            return True
        else:
            return False

class RandomLandmark(BaseLandmark):
    """
    Description: Used to simulate ramdom obstacles in the environment. 
                 These will not be consindered for A* path planning.
        
    """
    def __init__(self):
        super().__init__()
        self.size = np.array([0.2, 0.2])
    
    def is_collision(self, agent):
        x_min = self.state.p_pos[0] - self.size[0] / 2
        x_max = self.state.p_pos[0] + self.size[0] / 2
        y_min = self.state.p_pos[1] - self.size[1] / 2
        y_max = self.state.p_pos[1] + self.size[1] / 2
    
        agent_x = agent.state.p_pos[0]
        agent_y = agent.state.p_pos[1]
    
        return x_min <= agent_x <= x_max and y_min <= agent_y <= y_max
    
    def get_distance(self, agent_x, agent_y):
        dis_x = self.state.p_pos[0]-agent_x
        dis_y = self.state.p_pos[1]-agent_y
        return abs(dis_x), abs(dis_y)
    
class RectLandmark(BaseLandmark):
    """
    Description: Used to simulate the shelves in the environment.
                 These will be consindered for A* path planning.
        
    """
    def __init__(self):
        super().__init__()
        self.size = np.array([0.2, 0.4])
    
    def is_collision(self, agent):
        buffer = agent.size

        x_min = self.state.p_pos[0] - self.size[0] / 2 - buffer
        x_max = self.state.p_pos[0] + self.size[0] / 2 + buffer
        y_min = self.state.p_pos[1] - self.size[1] / 2 - buffer
        y_max = self.state.p_pos[1] + self.size[1] / 2 + buffer
    
        agent_x = agent.state.p_pos[0]
        agent_y = agent.state.p_pos[1]
    
        return x_min <= agent_x <= x_max and y_min <= agent_y <= y_max   
    
class World(BaseWorld):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    
    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            
            if p_force[i] is not None:
                entity.state.p_pos += p_force[i] * self.dt
                
            entity.state.p_vel = np.zeros_like(entity.state.p_vel)
    
    
    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                action = agent.action.u  # [0, -vx, 0, -vy, 0]
                vx = action[0]
                vy = action[1]
                scale = 2.0
                p_force[i] = np.array([vx, vy])* scale
        return p_force
    
    def apply_environment_force(self, p_force):
        return p_force
    
    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if entity_a is entity_b:
            return [None, None]  # don't collide against itsel
        
        # Rectangle to Rectangle collision
        if isinstance(entity_a, (RectLandmark, RandomLandmark)) and isinstance(entity_b, (RectLandmark, RandomLandmark)):
            delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
            dist_x = np.abs(delta_pos[0])  
            dist_y = np.abs(delta_pos[1])  

            half_width_a = entity_a.size[0] / 2
            half_height_a = entity_a.size[1] / 2
            half_width_b = entity_b.size[0] / 2
            half_height_b = entity_b.size[1] / 2
            
            dist_min_x = half_width_a + half_width_b
            dist_min_y = half_height_a + half_height_b
            
            # softmax penetration
            k = self.contact_margin
            
            penetration_x = np.logaddexp(0, -(dist_x - dist_min_x) / k) * k
            penetration_y = np.logaddexp(0, -(dist_y - dist_min_y) / k) * k
            
            if penetration_x > 0.01 and penetration_y > 0.01:
                if penetration_x > penetration_y:
                    force_magnitude = self.contact_force * penetration_x
                    force_direction = np.array([np.sign(delta_pos[0]), 0])
                else:
                    force_magnitude = self.contact_force * penetration_y
                    force_direction = np.array([0, np.sign(delta_pos[1])])
                    
                force = force_magnitude * force_direction
                force_a = force if entity_a.movable else None
                force_b = -force if entity_b.movable else None
                return [force_a, force_b]
            return [None, None]
        
        # Rectangle to Non-Rectangle collision
        elif isinstance(entity_a, (RectLandmark, RandomLandmark)) or isinstance(entity_b, (RectLandmark, RandomLandmark)):
            if isinstance(entity_a, (RectLandmark, RandomLandmark)):
                rect_entity = entity_a
                non_rect_entity = entity_b
            else: 
                rect_entity = entity_b
                non_rect_entity = entity_a
                
            delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
            dist_x = np.abs(delta_pos[0])  
            dist_y = np.abs(delta_pos[1])

            half_width = rect_entity.size[0] / 2
            half_height = rect_entity.size[1] / 2
            
            dist_min_x = half_width + non_rect_entity.size
            dist_min_y = half_height + non_rect_entity.size
            
            # softmax penetration
            k = self.contact_margin
            penetration_x = np.logaddexp(0, -(dist_x - dist_min_x) / k) * k
            penetration_y = np.logaddexp(0, -(dist_y - dist_min_y) / k) * k
            
            if penetration_x > 0.01 and penetration_y > 0.01:
                non_rect_entity.terminated = True
                if penetration_x > penetration_y:
                    penetration = penetration_x
                    direction = np.array([np.sign(delta_pos[0]), 0])
                else:
                    penetration = penetration_y
                    direction = np.array([0, np.sign(delta_pos[1])])

                force = self.contact_force * direction * penetration
                force_rect = -force if rect_entity.movable else None
                force_non_rect = force if non_rect_entity.movable else None
                
                if isinstance(entity_a, (RectLandmark, RandomLandmark)):
                    return [force_rect, force_non_rect]
                else: 
                    return [force_non_rect, force_rect]
                
            return [None, None]
        
        # Non-Rectangle to Non-Rectangle collision
        elif not isinstance(entity_a, (RectLandmark, RandomLandmark)) and not isinstance(entity_b, (RectLandmark, RandomLandmark)):        
            # compute actual distance between entities
            delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            # minimum allowable distance
            dist_min = entity_a.size + entity_b.size
            # softmax penetration
            k = self.contact_margin
            penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
            if penetration > 0.01:
                entity_a.terminated = True
                entity_b.terminated = True
            force = self.contact_force * delta_pos / dist * penetration
            force_a = +force if entity_a.movable else None
            force_b = -force if entity_b.movable else None
            return [force_a, force_b]


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_good=3,
        num_obstacles=4,
        max_cycles=100,
        continuous_actions=True,
        render_mode=None,
    ):
        EzPickle.__init__(
            self,
            num_good=num_good,
            num_obstacles=num_obstacles,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(num_good, num_obstacles)
        
        self.epsilon_runtime = scenario.epsilon_runtime
        self.epsilon_planning = scenario.epsilon_planning
        
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.metadata["name"] = "custom_environment"
        self.dynamic_agents = []
        size = 9
        self.state = [1] * (size * size)
        
    def draw(self):
        # clear screen
        self.screen.fill((255, 255, 255))

        # update bounds to center around agent
        all_poses = [entity.state.p_pos for entity in self.world.entities]
        cam_range = np.max(np.abs(np.array(all_poses)))

        # update geometry and text positions
        text_line = 0
        for e, entity in enumerate(self.world.entities):
            # geometry
            x, y = entity.state.p_pos
            y *= (
                -1
            )  # this makes the display mimic the old pyglet setup (ie. flips image)
            x = (
                (x / cam_range) * self.width // 2 * 0.9
            )  # the .9 is just to keep entities from appearing "too" out-of-bounds
            y = (y / cam_range) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2
            
            if isinstance(entity, RectLandmark):
                width, height = entity.size
                scale_factor = (self.width / (2 * cam_range)) * 0.9  # Match the scaling of positions
                rect_width = width * scale_factor
                rect_height = height * scale_factor
                pygame.draw.rect(
                    self.screen,
                    entity.color * 200,
                    pygame.Rect(
                        x - rect_width / 2,
                        y - rect_height / 2,
                        rect_width,
                        rect_height
                    )
                )
                pygame.draw.rect(
                    self.screen,
                    (0, 0, 0),  # Randfarbe (schwarz)
                    pygame.Rect(
                        x - rect_width / 2,
                        y - rect_height / 2,
                        rect_width,
                        rect_height
                    ), 
                    1  # Randdicke
                )
            elif isinstance(entity, RandomLandmark):
                width, height = entity.size
                scale_factor = (self.width / (2 * cam_range)) * 0.9  # Match the scaling of positions
                rect_width = width * scale_factor
                rect_height = height * scale_factor
                pygame.draw.rect(
                    self.screen,
                    entity.color * 200,
                    pygame.Rect(
                        x - rect_width / 2,
                        y - rect_height / 2,
                        rect_width,
                        rect_height
                    )
                )
                pygame.draw.rect(
                    self.screen,
                    (0, 0, 0),  # Randfarbe (schwarz)
                    pygame.Rect(
                        x - rect_width / 2,
                        y - rect_height / 2,
                        rect_width,
                        rect_height
                    ), 
                    1  # Randdicke
                )
            elif isinstance(entity, Agent):
                scale_factor = (self.width / (2 * cam_range)) * 0.9
                agent_radius = entity.size * scale_factor 
                
                pygame.draw.circle(
                    self.screen, entity.color * 200, (x, y), agent_radius
                )
                pygame.draw.circle(
                    self.screen, (0, 0, 0), (x, y), agent_radius, 1
                )
                # draw goal point
                goal_x, goal_y = entity.goal_point
                goal_y *= -1  # Flipping the y-axis

                goal_x = (goal_x / cam_range) * self.width // 2 * 0.9
                goal_y = (goal_y / cam_range) * self.height // 2 * 0.9
                goal_x += self.width // 2
                goal_y += self.height // 2
                pygame.draw.circle(
                    self.screen, entity.color * 200, (goal_x, goal_y), agent_radius, 6
                )
                # draw Q-Learning Area
                area_size = 9
                half_size = area_size//2
                cell_size = 0.05 * scale_factor
                for dx in range(-half_size, half_size+1):
                    for dy in range(-half_size, half_size+1):
                        if dx == 0 and dy == 0:
                            continue
                        
                        cell_x = x + dx * cell_size
                        cell_y = y + dy * cell_size
                        
                        pygame.draw.rect(
                            self.screen,
                            (100, 100, 255, 100),
                            pygame.Rect(
                                cell_x - cell_size / 2,
                                cell_y - cell_size / 2,
                                cell_size,
                                cell_size
                            ),
                            1
                        )
            
            assert (
                0 < x < self.width and 0 < y < self.height
            ), f"Coordinates {(x, y)} are out of bounds."
    
                
    def is_obstacle(self, x, y, q_learning = False, override_epsilon = None):
        """
        Description: checks if the given coordinates are colliding with obstacles.
        
        """
        epsilon = self.epsilon_planning if q_learning else self.epsilon_runtime
        if override_epsilon is not None:
            epsilon = override_epsilon
        # check for collision with wall
        border_limit = 0.98 
        if q_learning and (abs(x) >= border_limit or abs(y) >= border_limit):
            return True
        
        for landmark in self.world.landmarks:
            if isinstance(landmark, (RectLandmark, RandomLandmark) if q_learning else RectLandmark):
                x_min = landmark.state.p_pos[0] - landmark.size[0] / 2 - epsilon
                x_max = landmark.state.p_pos[0] + landmark.size[0] / 2 + epsilon
                y_min = landmark.state.p_pos[1] - landmark.size[1] / 2 - epsilon
                y_max = landmark.state.p_pos[1] + landmark.size[1] / 2 + epsilon
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    return True
        return False
    
    def discrete(self, agent_pos_x, agent_pos_y, goal_pos_x, goal_pos_y):
        """
        Description: Discretizing the environment.
        
        """
        cell_size = 0.05
        epsilon = 5e-2
        num_steps_x = int((1.95 // cell_size) + 1)
        num_steps_y = int((1.95 // cell_size) + 1)
        
        grid = []
        # 2D node list for adding children nodes
        node_grid = [[None for _ in range(num_steps_y)]for _ in range(num_steps_x)]
        start_node = None
        end_node = None
        min_start_dist = float('inf')
        min_end_dist = float('inf')
        
        # create nodes
        for x in range(num_steps_x):
            x_coordinate = -0.95 + x * cell_size
            for y in range(num_steps_y):
                y_coordinate = -0.95 + y * cell_size
                if not self.is_obstacle(x_coordinate, y_coordinate):
                    new_node = Node(x_coordinate, y_coordinate, 1)
                    node_grid[x][y] = new_node
                    grid.append(new_node)
                    
                    dist_start = math.hypot(x_coordinate - agent_pos_x, y_coordinate - agent_pos_y)
                    dist_end = math.hypot(x_coordinate - goal_pos_x, y_coordinate - goal_pos_y)

                    if dist_start < min_start_dist:
                        min_start_dist = dist_start
                        start_node = new_node

                    if dist_end < min_end_dist:
                        min_end_dist = dist_end
                        end_node = new_node
            
        # connect nodes
        for x in range(num_steps_x):
            for y in range(num_steps_y):
                node = node_grid[x][y]
                if node is not None:
                    # right
                    if x+1 < num_steps_x and node_grid[x + 1][y]:
                        node.add_child(node_grid[x+1][y])
                    # up
                    if y + 1 < num_steps_y and node_grid[x][y + 1]:
                        node.add_child(node_grid[x][y+1])
                    # left
                    if x - 1 >= 0 and node_grid[x - 1][y]:
                        node.add_child(node_grid[x-1][y])
                    # down
                    if y - 1 >= 0 and node_grid[x][y - 1]:
                        node.add_child(node_grid[x][y-1])

        if end_node is None:
                ("Endknoten konnte nicht gefunden werden")
        
        if start_node is None:
                ("Startknoten konnte nicht gefunden werden")
                            
        return start_node, end_node
    

    def heuristic(self, node, end_node):
        # euclidean distance
        #return ((node.x - end_node.x) ** 2 + (node.y - end_node.y) ** 2) ** 0.5
        return 0


    def A_star(self, start_node, end_node):
        """
        Description: returns the A*-path 
        
        """
        Open = [start_node]
        Closed = []
        
        start_node.cost_to_come = 0
        start_node.cost_to_go = self.heuristic(start_node, end_node)
        start_node.total_cost = start_node.cost_to_come + start_node.cost_to_go
        
        while Open:
            # best first search method with total cost
            current_node = min(Open, key=lambda node: node.total_cost)
            Open.remove(current_node)
            Closed.append(current_node)
            
            # check if end_node has been reached
            if (current_node.x, current_node.y) == (end_node.x, end_node.y):
                path = []
                while current_node is not None:
                    path.append((current_node.x, current_node.y))
                    current_node = current_node.parent
                return path[::-1]
            
            # check child nodes
            for child in current_node.children:
                if child in Closed:
                    continue

                # calculate new approx. cost
                approx_cost_to_come = current_node.cost_to_come + child.cost

                if child not in Open:
                    Open.append(child)
                elif approx_cost_to_come >= child.cost_to_come:
                    continue  

                child.parent = current_node
                child.cost_to_come = approx_cost_to_come
                child.cost_to_go = self.heuristic(child, end_node)
                child.total_cost = child.cost_to_come + child.cost_to_go
                
            if not Open:
                print("Hier läuft etwas schief")


    def is_dyn_obstacle(self, x, y, current_agent):
        """
        Description: checks if the given coordinates are colliding with an agent
                     returns either static or the direction the agent is moving in
        
        """
        for agent in self.world.agents:
            # agent should not detect himself as an obstacle
            if agent == current_agent:
                continue
            else:
                euclidian_dis = math.sqrt((x - agent.state.p_pos[0]) ** 2 + (y - agent.state.p_pos[1]) ** 2)
                if euclidian_dis <= agent.size:
                    if agent.action.u is None:
                        return "static"
                    
                    vx = agent.action.u[0]
                    vy = agent.action.u[1]
                    
                    if abs(vx) < 1e-3 and abs(vy) < 1e-3:
                        return "static"
                    else:
                        if abs(vx) > abs(vy):
                            return "east" if vx > 0 else "west"
                        else:
                            return "north" if vy > 0 else "south"
        return "free"
    
    
    def q_learning_state_space(self, agent, a_star_new, a_star_old):
        """
        Description: returns the Q-Learning state space
        
        STATUS MAPPING:
            free            = 1
            static obstacle = 2 #cannot ever be part of astar path
            static agent    = 3
            moving north    = 4
            moving south    = 5
            moving west     = 6
            moving east     = 7
            unknown         = 8 #cannot ever be part of astar path
            Astar new       =+8
            Astar old       =+16
        """
        size = 9
        agent_grid = 4
        cell_size = 0.05
        agent_pos_x = agent.state.p_pos[0]
        agent_pos_y = agent.state.p_pos[1]
        start_pos_x = agent_pos_x - agent_grid*cell_size
        start_pos_y = agent_pos_y - agent_grid*cell_size

        # set all cells to free
        state = [1]*(size * size)
        
        def get_index(r, c):
            return c * size + r
        
        for r in range(size):
            for c in range(size):
                index = get_index(r,c)
                x, y = start_pos_x + c * cell_size, start_pos_y + r * cell_size
                if(r, c) == (agent_grid, agent_grid):
                    state[index] = 1
                
                elif self.is_obstacle(x, y, q_learning=True, override_epsilon=0.015):
                    state[index] = 2
                else:
                    dyn_status = self.is_dyn_obstacle(x, y, agent)
                    if dyn_status == "static":
                        state[index] =3
                    elif dyn_status == "north":
                        state[index] = 4
                    elif dyn_status == "south":
                        state[index] = 5
                    elif dyn_status == "west":
                        state[index] = 6
                    elif dyn_status == "east":
                        state[index] = 7

        epsilon = 25e-3 
        for r in range(size):
            for c in range(size):
                index = get_index(r,c)
                # static obstacle(2) can also be part of the Astar path -> random obstacles, overlapping
                if state[index] in [1, 2, 3, 4, 5, 6, 7]:
                    x, y = start_pos_x + c * cell_size, start_pos_y + r * cell_size
                    for path_x, path_y in a_star_new:
                        if math.isclose(x, path_x, abs_tol=epsilon) and math.isclose(y, path_y, abs_tol=epsilon):
                            state[index] += 8
                    for path_x, path_y in a_star_old:        
                        if math.isclose(x, path_x, abs_tol=epsilon) and math.isclose(y, path_y, abs_tol=epsilon):
                            state[index] += 16                  
        return tuple(state)

    def last(
        self, observe: bool = True
    ) -> tuple[ObsType | None, float, bool, bool, dict[str, Any]]:
        """Returns observation, cumulative reward, terminated, truncated, info for the current agent (specified by self.agent_selection)."""
        agent = self.agent_selection # current agent that is being stepped
        assert agent is not None
        observation = self.observe(agent) if observe else None
        agent_object = self.world.agents[self._index_map[agent]]
        
        agent_object.q_state = self.q_learning_state_space(agent_object, agent_object.a_star_new, agent_object.a_star_old)
        # check for +8 and +16 A_star flag
        if not any((val & 8) or (val & 16) for val in agent_object.q_state):
            self.terminations[agent] = True
            agent_object.terminated = True
            print(f"[TERMINATED] {agent} sees no A* path in local state — terminating.")
        
        return (
            observation,
            self._cumulative_rewards[agent],
            self.terminations[agent],
            self.truncations[agent],
            self.infos[agent],
            agent_object.q_state
        )


    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed(seed=seed)
        self.scenario.reset_world(self.world, self.np_random)

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0

        self.current_actions = [None] * self.num_agents
        
        agent_states ={}
        agent_observations ={}
        
        for agent in self.agents:
            assert agent is not None
            observation = self.observe(agent)
            start_node, end_node = self.discrete(observation[0], observation[1], observation[2], observation[3])
            agent_object = self.world.agents[self._index_map[agent]]
            a_star_path = self.A_star(start_node, end_node)
            if a_star_path is None:
                print("Es konnte kein Pfad gefunden werden")
            agent_object.a_star_new = a_star_path[2:] # remove first two elements
            agent_object.a_star_old = []
            agent_object.q_state = self.q_learning_state_space(agent_object, agent_object.a_star_new, agent_object.a_star_old)
            agent_object.movable = True
            agent_states[agent] = agent_object.q_state
            agent_observations[agent] = observation
            agent_object.controller_x = PID(setpoint=0)
            agent_object.controller_y = PID(setpoint=0)
            agent_object.terminated = False
        return agent_states, agent_observations
    
        
    def _skip_dead_agent(self):
        agent = self.agent_selection
        agent_obj = next((a for a in self.world.agents if a.name == agent), None)
        agent_obj.movable = False
        
        agent_list = list(self.agents)
        if all(self.terminations[agent] or self.truncations[agent] for agent in agent_list):
            return 
        
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()
        
        if next_idx == 0:
            self._execute_world_step()
            self.steps += 1
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.truncations[a] = True
        
        
        
        
    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._skip_dead_agent()
            return
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action

        if next_idx == 0:
            self._execute_world_step()
            self.steps += 1
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.truncations[a] = True
        else:
            self._clear_rewards()

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

    def next_agent(self):
        self.agent_selection = self._agent_selector.next()
        
    
    def was_astar_step(self, agent):
        epsilon = 25e-3
        for path_x, path_y in agent.a_star_new:
            if math.isclose(agent.state.p_pos[0],path_x, abs_tol=epsilon) and math.isclose(agent.state.p_pos[1], path_y, abs_tol=epsilon):
                print_state(agent.q_state)
                path_node = (path_x, path_y)
                agent.a_star_old.append(path_node)
                agent.a_star_new.remove(path_node)
                return True
        return False
    

    def _execute_world_step(self):
        # set action for each agent
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            if agent.movable:
                mdim = self.world.dim_p * 2 + 1
                if self.continuous_actions:
                    scenario_action.append(action[0:mdim])
                    action = action[mdim:]
                else:
                    scenario_action.append(action % mdim)
                    action //= mdim
            if not agent.silent:
                scenario_action.append(action)
            self._set_action(scenario_action, agent, self.action_spaces[agent.name])

        self.world.step()

        global_reward = 0.0
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_reward(self.world))
        

        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world, self.was_astar_step(agent)))
            if self.local_ratio is not None:
                reward = (
                    global_reward * (1 - self.local_ratio)
                    + agent_reward * self.local_ratio
                )
            else:
                reward = agent_reward

            self.rewards[agent.name] = reward
            self.terminations[agent.name] = self.scenario.is_termination(agent, self.world)
            
    def get_next_point(self, agent_x, agent_y, action, agent):
        action_print = "nothing"
        gridsize = 0.05
        goal_point = [0.0, 0.0]
        if action == 0: #up
            goal_point[0]=agent_x
            goal_point[1]=agent_y+gridsize
            action_print = "up"
        if action == 1: #down
            goal_point[0]=agent_x
            goal_point[1]=agent_y-gridsize
            action_print = "down"
        if action == 2: #left
            goal_point[0]=agent_x-gridsize
            goal_point[1]=agent_y
            action_print = "left"
        if action == 3: #right
            goal_point[0]=agent_x+gridsize
            goal_point[1]=agent_y   
            action_print = "right"
        if action == 4: #wait
            goal_point[0]=agent_x
            goal_point[1]=agent_y
            action_print = "wait"
        
        if agent == None:
            return goal_point
        else:
            #print(agent + ": Angesteuert wird Punkt: " + str(goal_point[0]) + ", " + str(goal_point[1]) + " . Von Punkt: "+str(agent_x) + " " + str(agent_y) + " . Aktion: " + action_print)
            return goal_point

    def get_cont_action(self, observation, dimension, discrete_action, agent):
        agent_object = self.world.agents[self._index_map[agent]]
        next_point = self.get_next_point(observation[0], observation[1], discrete_action, agent)
        
        if discrete_action == 4:  # WAIT
            agent_object.controller_x.reset()
            agent_object.controller_y.reset()
            return np.zeros(dimension * 2 + 1)
        
        agent_object.controller_x.setpoint = next_point[0]
        agent_object.controller_y.setpoint = next_point[1]
        
        agent_object.state.p_vel = np.zeros_like(agent_object.state.p_vel)
        
        v_x = agent_object.controller_x(observation[0])
        v_y = agent_object.controller_y(observation[1])
        
        action = np.zeros(dimension * 2 + 1)
        action[1] = -v_x
        action[2] = 0
        action[3] = -v_y
        action[4] = 0
        #print(agent + ": vx: " + str(v_x) + " . vy: " + Fstr(v_y)) 
        return action


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.epsilon_runtime = 5e-3
        self.epsilon_planning = 9e-2
    
    def make_world(self, num_good=2, num_obstacles=4):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = num_good
        num_landmarks = num_obstacles
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            base_name =  "agent"
            base_index = i
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
            agent.accel = 4.0
            agent.max_speed = 1.3
            agent.damping = 0.5
        # add landmarks
        world.landmarks = [RectLandmark() for i in range(num_landmarks)]
        world.landmarks.append(RandomLandmark())
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = np.array([0.2, 0.4])
            landmark.boundary = False
        world.state = 0
        return world


    def reset_world(self, world, np_random):
        colors = [
            [0.0, 1.0, 0.0],  # green
            [0.0, 0.0, 1.0],  # blue
            [1.0, 1.0, 0.0],  # yellow
            [1.0, 0.5, 0.0],  # orange
            [0.5, 0.0, 0.5],  # purple
            [0.0, 1.0, 1.0],  # cyan
            [1.0, 0.0, 1.0],  # magenta
        ]
        
        # set states for landmarks
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_vel = np.zeros(world.dim_p)
                landmark.color = np.array([0.25, 0.25, 0.25])
        
        # set initial states for rectangle landmarks
        world.landmarks[0].state.p_pos= np.array([-0.25, -0.3])
        world.landmarks[1].state.p_pos= np.array([-0.25, 0.3])
        world.landmarks[2].state.p_pos= np.array([0.25, -0.3])
        world.landmarks[3].state.p_pos= np.array([0.25, 0.3])
        
        # set random intial state for landmark
        world.landmarks[4].state.p_pos= np_random.uniform(-1, +1, world.dim_p)
        world.landmarks[4].color = np.array([1.0, 0.0, 0.0])
        world.landmarks[4].size = np.array([0.15, 0.15])
        
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array(colors[i % len(colors)])
            )
        
        for agent in world.agents:
            agent.state.p_vel = np.zeros(world.dim_p)
            
            while True:
                pos = np_random.uniform(-1.0, +1.0, world.dim_p)
                if self.is_in_landmark(world, pos[0], pos[1]):
                    continue
                
                agent.state.p_pos = pos
                
                collision = any(
                    np.linalg.norm(agent.state.p_pos - other.state.p_pos) < (agent.size + other.size)
                    for other in world.agents 
                    if other is not agent and other.state.p_pos is not None
                )
                
                if not collision:
                    break
                
                agent.state.p_pos = None
                
            agent.state.c = np.zeros(world.dim_c)
            while True:
                pos = np_random.uniform(-0.6, +0.6, world.dim_p)
                if self.is_in_landmark(world, pos[0], pos[1]):
                    continue
                goal_collision = any(
                    hasattr(other, 'goal_point') and isinstance(other.goal_point, list) and len(other.goal_point) == 2 and
                    np.linalg.norm(np.array(pos) - np.array(other.goal_point)) < 0.15
                    for other in world.agents 
                    if other is not agent
                )
            
                if not goal_collision:
                    agent.goal_point = pos
                    break
            

        
    def is_in_landmark(self, world, pos_x, pos_y):
        for landmark in world.landmarks:
            if isinstance(landmark, (RectLandmark, RandomLandmark)):
                if landmark.state.p_pos is None:
                    continue
                x_min = landmark.state.p_pos[0] - landmark.size[0] / 2 - self.epsilon_planning
                x_max = landmark.state.p_pos[0] + landmark.size[0] / 2 + self.epsilon_planning
                y_min = landmark.state.p_pos[1] - landmark.size[1] / 2 - self.epsilon_planning
                y_max = landmark.state.p_pos[1] + landmark.size[1] / 2 + self.epsilon_planning
    
                if x_min <= pos_x <= x_max and y_min <= pos_y <= y_max:
                        return True
        return False
        
        
    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size + self.epsilon_runtime
        return True if dist < dist_min else False
    
    def is_out_of_bounds(self, agent, margin = 0.05):
        bound = 1.0 + margin
        return np.any(np.abs(agent.state.p_pos) > bound)
    
    def is_goal(self, agent):
        delta_pos = agent.state.p_pos - agent.goal_point
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent.size + self.epsilon_runtime
        return True if dist < dist_min else False  
            
    
    def reward(self, agent, world, was_astar_step):
        reward = 0
        for landmark in world.landmarks:
            if landmark.is_collision(agent):
                reward -= 1    # collision
                    
        for other_agent in world.agents:
            if self.is_collision(agent, other_agent):
                if other_agent == agent:
                    continue
                else:
                    reward -= 1    # collision
        
        for other_agent in world.agents:
            if self.is_out_of_bounds(agent):
                reward -= 1         # collision
                
        if was_astar_step:
            reward += 0.5        # choose A*
            
        reward -= 0.1          # time penalty
                
        return reward

    def agents(self, world):
        return world.agents


    def observation(self, agent, world):
        # world argument is kept because methods in the simple_env.py have it based on the original version of the observation method
        agent_pos = np.array(agent.state.p_pos)
        goal_pos = np.array(agent.goal_point)
        return np.concatenate((agent_pos, goal_pos))

    
    def is_termination(self, agent, world):
        if agent.terminated:
            return True
        
        termination = False
        for landmark in world.landmarks:
            if landmark.is_collision(agent):
                termination = True
                    
        for other_agent in world.agents:
            if self.is_collision(agent, other_agent):
                if other_agent == agent:
                    continue
                else:
                    termination = True
        
        if self.is_out_of_bounds(agent):
            termination = True
            
        if self.is_goal(agent):
            termination = True
        
        return termination
