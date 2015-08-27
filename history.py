class History:
    def __init__(self):
        self.history = []

    def new_episode(self):
        """Initialize new episode"""
        self.history.append([])

    def record(self, action, observation):
        """Record action and subsequent observation"""
        self.history[-1].append((action, observation))

    def n_episodes(self):
        return len(self.history)

    def n_t(self, episode):
        """Return number of actions taken in episode"""
        return len(self.history[episode])

    def get_AO(self, episode, worker_separator=None):
        """Get action and observation pairs for the given episode.

        Args:
            episode (int):              Episode number. 
            worker_separator (tuple):   (action, observation) that separates
                                        workers. Either action or observation
                                        may be null.

        Returns:
            List of (action, observation) pairs, if worker_separator is None.
            Else, list of such lists for each worker.

        """
        if worker_separator is None:
            return self.history[episode]
        else:
            def separator_match(tup):
                return all(v is None or v == v_t for v, v_t in
                           zip(worker_separator, tup))
            workers = [[]]
            for x in self.history[episode]:
                workers[-1].append(x)
                if separator_match(x):
                    workers.append([])
            return workers
