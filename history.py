class History:
    def __init__(self):
        self.history = []

    def new_worker(self):
        """Initialize new worker"""
        self.history.append([])

    def record(self, action, observation):
        """Record action and subsequent observation"""
        self.history[-1].append((action, observation))

    def n_workers(self):
        return len(self.history)

    def n_t(self, worker):
        """Return number of actions taken with worker"""
        return len(self.history[worker])
