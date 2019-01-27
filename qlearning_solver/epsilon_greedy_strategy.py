 ###############################################################################
 # Copyright (C) 2019 Charly Lamothe, Guillaume Ollier                         #
 #                                                                             #
 # This file is part of QLearningSolver.                                       #
 #                                                                             #
 #   Licensed under the Apache License, Version 2.0 (the "License");           #
 #   you may not use this file except in compliance with the License.          #
 #   You may obtain a copy of the License at                                   #
 #                                                                             #
 #   http://www.apache.org/licenses/LICENSE-2.0                                #
 #                                                                             #
 #   Unless required by applicable law or agreed to in writing, software       #
 #   distributed under the License is distributed on an "AS IS" BASIS,         #
 #   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  #
 #   See the License for the specific language governing permissions and       #
 #   limitations under the License.                                            #
 ###############################################################################

import random
import numpy as np


class EpsilonGreedyStrategy(object):
    """
    This strategy allows for a compromise between exploration
    and exploration. If the variable drawn randomly using a
    uniform law is lesser than a specified epsilon, we choose
    an action randomly, in order to explore. Otherwise, we compute
    the action who maximizes the couple state-action.
    """

    def __init__(self, epsilon=0.1, decrease_factor=0.999):
        self._epsilon = epsilon  # Exploration factor
        self._decrease_factor = decrease_factor # Decrease factor of epsilon

    def choose_action(self, state, actions, rewards):
        if np.random.uniform(0, 1) < self._epsilon:
            action = random.choice(actions)
        else:
            best_reward = max(rewards)

            """
            In the case where there are multiple state-action couples
            with the same value, we choose one randomly among them.
            """
            if rewards.count(best_reward) > 1:
                best_actions = [actions[i] for i in range(len(actions)) if rewards[i] == best_reward]
                action = random.choice(best_actions)
            else:
                action = actions[rewards.index(best_reward)]

        return action

    def update(self):
        self._epsilon *= self._epsilon * self._decrease_factor
