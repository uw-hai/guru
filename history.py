class History:
    def __init__(self):
        self.history = {'actions': [], 'observations': []}

    def new_episode(self):
        """Initialize new episode"""
        self.history['actions'].append([])
        self.history['observations'].append([])

    def record(self, action, observation):
        """Record action and subsequent observation"""
        self.history['actions'][-1].append(action)
        self.history['observations'][-1].append(observation)

    def n_episodes(self):
        return len(self.history['actions'])

    def n_t(self, episode):
        """Return number of actions taken in episode"""
        return len(self.history['actions'][episode])

    def get_AO(self, episode, t):
        """Get action and subsequent observation"""
        return (self.history['actions'][episode][t],
                self.history['observations'][episode][t])

    def get_actions(self, episode):
        """Get list of actions for episode"""
        return self.history['actions'][episode]

    def get_observations(self, episode):
        """Get list of observations for episode"""
        return self.history['observations'][episode]
