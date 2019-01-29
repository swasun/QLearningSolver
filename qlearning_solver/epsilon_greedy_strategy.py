 #####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (C) 2018 Charly Lamothe, Guillaume Ollier                               #
 #                                                                                   #
 # This file is part of QLearningSolver.                                             #
 #                                                                                   #
 #   Permission is hereby granted, free of charge, to any person obtaining a copy    #
 #   of this software and associated documentation files (the "Software"), to deal   #
 #   in the Software without restriction, including without limitation the rights    #
 #   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell       #
 #   copies of the Software, and to permit persons to whom the Software is           #
 #   furnished to do so, subject to the following conditions:                        #
 #                                                                                   #
 #   The above copyright notice and this permission notice shall be included in all  #
 #   copies or substantial portions of the Software.                                 #
 #                                                                                   #
 #   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR      #
 #   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        #
 #   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE     #
 #   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER          #
 #   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   #
 #   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE   #
 #   SOFTWARE.                                                                       #
 #####################################################################################

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
