# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        for i in range(self.iterations):
            new_values = util.Counter()
            states = self.mdp.getStates()

            for state in states:
                if self.mdp.isTerminal(state):
                    new_values[state] = 0
                else:
                    actions = self.mdp.getPossibleActions(state)
                    if len(actions) > 0:
                        max_q = max([self.computeQValueFromValues(state, action) for action in actions])
                        new_values[state] = max_q
            self.values = new_values


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        q_value = 0
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)

        for nextState, prob in transitions:
            reward = self.mdp.getReward(state, action, nextState)
            q_value += prob * (reward + self.discount * self.values[nextState])

        return q_value

    def computeActionFromValues(self, state):
        if self.mdp.isTerminal(state):
            return None

        actions = self.mdp.getPossibleActions(state)
        best_action = None
        max_q = float('-inf')

        for action in actions:
            q_val = self.computeQValueFromValues(state, action)
            if q_val > max_q:
                max_q = q_val
                best_action = action

        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

    def getValue(self, state):
        return self.values[state]


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()

        # 1. Calculăm predecesorii pentru toate stările
        # Predecesorii unei stări s sunt toate stările s' din care se poate ajunge în s cu probabilitate > 0
        predecessors = {}
        for s in states:
            for action in self.mdp.getPossibleActions(s):
                for nextState, prob in self.mdp.getTransitionStatesAndProbs(s, action):
                    if prob > 0:
                        if nextState not in predecessors:
                            predecessors[nextState] = set()
                        predecessors[nextState].add(s)

        # 2. Inițializăm coada de priorități goală
        pq = util.PriorityQueue()

        # 3. Pentru fiecare stare non-terminală s:
        for s in states:
            if not self.mdp.isTerminal(s):
                # Găsim valoarea maximă a Q(s, a)
                max_q = max([self.computeQValueFromValues(s, a) for a in self.mdp.getPossibleActions(s)])
                diff = abs(self.values[s] - max_q)
                # Punem s în PQ cu prioritatea -diff (deoarece PriorityQueue din util.py scoate elementul cu valoarea cea mai mică)
                pq.update(s, -diff)

        # 4. Rulăm numărul de iterații specificat
        for i in range(self.iterations):
            if pq.isEmpty():
                break

            # Scoatem starea s cu cea mai mare eroare
            s = pq.pop()

            # Actualizăm valoarea stării s (dacă nu e terminală)
            if not self.mdp.isTerminal(s):
                max_q = max([self.computeQValueFromValues(s, a) for a in self.mdp.getPossibleActions(s)])
                self.values[s] = max_q

            # Pentru fiecare predecesor p al lui s:
            if s in predecessors:
                for p in predecessors[s]:
                    # Calculăm diff pentru predecesor
                    max_q_p = max([self.computeQValueFromValues(p, a) for a in self.mdp.getPossibleActions(p)])
                    diff = abs(self.values[p] - max_q_p)

                    # Dacă eroarea este mai mare decât theta, actualizăm prioritatea în PQ
                    if diff > self.theta:
                        pq.update(p, -diff)
