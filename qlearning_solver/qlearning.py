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

class QLearning(object):
    """
    Q(s, a) += alpha * (reward(s, a) + gamma * max(Q(s', a') - Q(s, a))
    with :
        - alpha the learning rate, which module the learning speed,
        - gamme the actualisation factor, which quantifies the importance of the next action.
    
    We will update the value of Q(s, a) with the actually obtained reward
    from the state s using the action a, to which we will add the best reward
    we can get in the future.
    """

    def __init__(self, actions, strategy, alpha=0.1, gamma=0.9, decrease_alpha=True, decrease_alpha_min=0.005, decrease_alpha_factor=0.85):
        self._actions = actions  # Action list
        self._strategy = strategy # Action choice strategy used by the algorithm
        self._alpha = alpha # Learning rate
        self._gamma = gamma # Actualisation factor
        self._decrease_alpha = decrease_alpha # Reduce the learning rate progressively
        self._decrease_alpha_min = decrease_alpha_min # Min value of the learning rate if we reduce it
        self._decrease_alpha_factor = decrease_alpha_factor # Factor used to decrease the learning rate if the option is used
        self._q = {} # Q-value table

    def _get_reward(self, state, action):
        """
        Compute the reward of an a action at a certain state.
        The default reward is 0.0.
        """
        return self._q.get((state, action), 0.0)

    def _get_rewards(self, state, actions):
        """
        Compute the rewards of all state-action couples.
        """
        return [self._get_reward(state, a) for a in actions]

    def choose_action(self, state):
        """
        Chooses an action using the specified strategy.
        """
        return self._strategy.choose_action(state, self._actions, self._get_rewards(state, self._actions))

    def learn(self, current_state, action, new_state, reward, episode):
        """
        If we didn't know a reward for this configuration, we update
        the couple with the reward with just win, otherwise we update
        using the table Q-value using the following formula:
        Q(s, a) += alpha * (reward(s, a) + gamma * max(Q(s', a') - Q(s, a))
        """
        previous_reward = self._q.get((current_state, action), None)
        if previous_reward is None:
            self._q[(current_state, action)] = reward
        else:
            """
            Update the Q-value table with:
                - previous_reward the best previous reward until now,
                - self._alpha the learning rate which module the learning speep,
                - next_best_reward the estimated best reward,
                - reward + self._gamma * next_best_reward the learned value.
            """
            next_best_reward = max(self._get_rewards(new_state, self._actions))
            eta = max(self._decrease_alpha_min, 1.0 * (self._decrease_alpha_factor ** (episode // 100))) if self._decrease_alpha else self._alpha
            self._q[(current_state, action)] = previous_reward + eta * (reward + self._gamma * next_best_reward - previous_reward)
