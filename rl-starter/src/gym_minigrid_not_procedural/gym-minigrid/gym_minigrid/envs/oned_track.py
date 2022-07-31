from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class OneDTrackEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(3,13),
        agent_start_dir=3,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            width=7,
            height=16,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        #self.put_obj(Goal(), width - 2, height-2)

        for j in range(height-2):
            self.put_obj(Wall(), width - 3, j+1)
            if j==3:
                is_locked=False
                self.put_obj(Goal(), width - 6, j+1)
            else:
                is_locked=True
            if j==10:
                self.put_obj(Ball('yellow'), width - 2, j+1)
            if j==6:
                self.put_obj(Ball('red'), width - 2, j+1)
            if j==2:
                self.put_obj(Ball('purple'), width -2, j+1)
            if j==5:
                self.put_obj(Key('yellow'), width -2, j+1)
            self.put_obj(Door('blue', is_locked=is_locked), width-5,  j+1)

        if self.step_count == 0:
            self.place_random_tiles()


        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"


    def place_random_tiles(self):
        for i in range(self.grid.height-1):
            for j in range(self.grid.width-1):
                if j == 3:
                    continue
                if self.grid.get(j, i) == None or type(self.grid.get(j, i)) == RandomTile:
                    self.put_obj(RandomTile(), j, i)

    def step(self, action):
        self.step_count += 1

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = self._reward()
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        self.place_random_tiles()

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}

class OneDTrackEnv7x7(OneDTrackEnv):
    def __init__(self, **kwargs):
        super().__init__(size=7, **kwargs)

register(
    id='MiniGrid-OneD-7x7-v0',
    entry_point='gym_minigrid.envs:OneDTrackEnv7x7'
)
