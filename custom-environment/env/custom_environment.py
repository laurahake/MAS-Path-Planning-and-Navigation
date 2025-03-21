import numpy as np
import pygame
import math
import string
from node_class import Node
from gymnasium.utils import EzPickle
from gymnasium.core import ObsType
from typing import Any

from pettingzoo.mpe._mpe_utils.core import World
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
    def __init__(self):
        super().__init__()
        self.a_star_old = []
        self.a_star_new = []
        self.goal_point = []
        size = 9
        self.q_state = [1] * (size * size)


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
        x_min = self.state.p_pos[0] - self.size[0] / 2
        x_max = self.state.p_pos[0] + self.size[0] / 2
        y_min = self.state.p_pos[1] - self.size[1] / 2
        y_max = self.state.p_pos[1] + self.size[1] / 2
    
        agent_x = agent.state.p_pos[0]
        agent_y = agent.state.p_pos[1]
    
        return x_min <= agent_x <= x_max and y_min <= agent_y <= y_max   


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
    
                
    def is_obstacle(self, x, y, q_learning = False):
        """
        Description: checks if the given coordinates are colliding with obstacles.
        
        """
        for landmark in self.world.landmarks:
            if isinstance(landmark, (RectLandmark, RandomLandmark) if q_learning else RectLandmark):
                x_min = (landmark.state.p_pos[0] - landmark.size[0] / 2) -0.05
                x_max = (landmark.state.p_pos[0] + landmark.size[0] / 2) +0.05
                y_min = (landmark.state.p_pos[1] - landmark.size[1] / 2) -0.05
                y_max = (landmark.state.p_pos[1] + landmark.size[1] / 2) +0.05
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
        
        # create nodes
        for x in range(num_steps_x):
            x_coordinate = -0.95 + x * cell_size
            for y in range(num_steps_y):
                y_coordinate = -0.95 + y * cell_size
                if not self.is_obstacle(x_coordinate, y_coordinate):
                    new_node = Node(x_coordinate, y_coordinate, 1)
                    node_grid[x][y] = new_node
                    grid.append(new_node)
                    if math.isclose(x_coordinate, agent_pos_x, abs_tol=epsilon) and math.isclose(y_coordinate, agent_pos_y, abs_tol=epsilon):
                        start_node = node_grid[x][y]
                    if math.isclose(x_coordinate, goal_pos_x, abs_tol=epsilon) and math.isclose(y_coordinate, goal_pos_y, abs_tol=epsilon):
                        end_node = node_grid[x][y]
                
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
        status = "free"
        for agent in self.world.agents:
            # agent should not detect himself as an obstacle
            if agent == current_agent:
                status = "free"
            else:
                euclidian_dis = math.sqrt((x - agent.state.p_pos[0]) ** 2 + (y - agent.state.p_pos[1]) ** 2)
                if euclidian_dis <= agent.size:
                    
                    if agent.state.p_vel[0] == 0 and agent.state.p_vel[1] == 0:
                        return "static"
                    else:
                        if abs(agent.state.p_vel[0]) > abs(agent.state.p_vel[1]):
                            if agent.state.p_vel[0] > 0:
                                status = "east"
                            else:
                                status = "west"
                        else:
                            if agent.state.p_vel[1] > 0:
                                status = "north"
                            else:
                                status = "south"
        
        return status
    
    
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
                
                elif self.is_obstacle(x, y, q_learning=True):
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

        # set spaces behind obstacles to unkown
        for r in range(size):
            for c in range(size):
                index = get_index(r, c)
                
                if state[index] in [2, 3, 4, 5, 6, 7]:
                    # find direction relative to agent
                    dr = r - agent_grid
                    dc = c - agent_grid
                    if abs(dr) > abs(dc):
                        # up or down
                        direction = (-1, 0) if dr < 0 else (1, 0)
                    elif abs(dr) < abs(dc):
                        # left or right
                        direction = (0, -1) if dc < 0 else (0, 1)
                    else:
                        # diagonal
                        if dr == dc:
                            direction = (-1, -1) if dc <0 else (1, 1) # south-west or north-east
                        else:
                            direction = (1, -1) if dc < dr else (-1, 1) # north-west or south-east

                    nr = r + direction[0]
                    nc = c + direction[1]
                    while 0 <= nr < size and 0 <= nc < size:
                        next_index = get_index(nr, nc)
                        if state[next_index] in [2, 3, 4, 5, 6, 7]:
                            break
                        state[next_index] = 8
                        nr = int(nr + direction[0])
                        nc = int(nc + direction[1])

        epsilon = 25e-3 
        for r in range(size):
            for c in range(size):
                index = get_index(r,c)
                if state[index] in [1, 3, 4, 5, 6, 7]:
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
            agent_object.a_star_new = a_star_path[1:] # remove first element
            agent_object.a_star_old = []
            agent_object.q_state = self.q_learning_state_space(agent_object, agent_object.a_star_new, agent_object.a_star_old)
            agent_object.movable = True
            agent_states[agent] = agent_object.q_state
            agent_observations[agent] = observation
            
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

env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
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
                    self.is_collision(agent, other) 
                    for other in world.agents 
                    if other is not agent and other.state.p_pos is not None
                )
                
                if not collision:
                    break
                
                agent.state.p_pos = None
                
            agent.state.c = np.zeros(world.dim_c)
            while True:
                pos = np_random.uniform(-0.6, +0.6, world.dim_p)
                if not self.is_in_landmark(world, pos[0], pos[1]):
                    agent.goal_point = pos
                    break
            

        
    def is_in_landmark(self, world, pos_x, pos_y):
        for landmark in world.landmarks:
            if isinstance(landmark, RectLandmark):
                x_min = (landmark.state.p_pos[0] - landmark.size[0] / 2) -0.05
                x_max = (landmark.state.p_pos[0] + landmark.size[0] / 2) +0.05
                y_min = (landmark.state.p_pos[1] - landmark.size[1] / 2) -0.05
                y_max = (landmark.state.p_pos[1] + landmark.size[1] / 2) +0.05
    
                if x_min <= pos_x <= x_max and y_min <= pos_y <= y_max:
                        return True
        return False
        
        
    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False
    
    def is_out_of_bounds(self, agent):
        x = agent.state.p_pos[0]
        y = agent.state.p_pos[1]
        epsilon = 25e-3
        if math.isclose(x, 1.0, abs_tol=epsilon) or math.isclose(x, -1.0, abs_tol=epsilon):
            return True
        elif math.isclose(y, 1.0, abs_tol=epsilon) or math.isclose(y, -1.0, abs_tol=epsilon):
            return True
            
    
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
            
        reward -= 0.01          # time penalty
                
        return reward

    def agents(self, world):
        return world.agents


    def observation(self, agent, world):
        # world argument is kept because methods in the simple_env.py have it based on the original version of the observation method
        agent_pos = np.array(agent.state.p_pos)
        goal_pos = np.array(agent.goal_point)
        return np.concatenate((agent_pos, goal_pos))

    
    def is_termination(self, agent, world):
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
        
        return termination
