# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minimax(state, depth):
            if (state.isWin() or state.isLose()) or depth == 0:
                return None

            legal_actions = state.getLegalActions(0)
            score = float('-inf')
            action = None
            for cur_action in legal_actions:
                next_state = state.generateSuccessor(0, cur_action)
                cur_score = recursiveMinimax(next_state, depth, 1)
                if cur_score > score:
                    score = cur_score
                    action = cur_action
            return action

        def recursiveMinimax(state, depth, agent):
            if (state.isWin() or state.isLose()) or depth == 0:
                return self.evaluationFunction(state)

            legal_actions = state.getLegalActions(agent)
            next_agent = (agent + 1) % state.getNumAgents()

            if agent == 0:
                score = float('-inf')
                for cur_action in legal_actions:
                    next_state = state.generateSuccessor(agent, cur_action)
                    score = max(score, recursiveMinimax(next_state, depth, next_agent))
            else:
                score = float('inf')
                for cur_action in legal_actions:
                    next_state = state.generateSuccessor(agent, cur_action)
                    if next_agent == 0:
                        score = min(score, recursiveMinimax(next_state, depth - 1, next_agent))
                    else:
                        score = min(score, recursiveMinimax(next_state, depth, next_agent))
            return score

        return minimax(gameState, self.depth)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphaBeta(state, depth):
            if (state.isWin() or state.isLose()) or depth == 0:
                return None

            alpha = float('-inf')
            beta = float('inf')
            legal_actions = state.getLegalActions(0)

            score = float('-inf')
            action = None
            for cur_action in legal_actions:
                next_state = state.generateSuccessor(0, cur_action)
                cur_score = recursiveAlphaBeta(next_state, depth, 1, alpha, beta)
                if cur_score > score:
                    score = cur_score
                    action = cur_action
                if score > beta:
                    break
                alpha = max(alpha, score)
            return action

        def recursiveAlphaBeta(state, depth, agent, alpha, beta):
            if (state.isWin() or state.isLose()) or depth == 0:
                return self.evaluationFunction(state)

            legal_actions = state.getLegalActions(agent)
            next_agent = (agent + 1) % state.getNumAgents()

            if agent == 0:
                score = float('-inf')
                for cur_action in legal_actions:
                    next_state = state.generateSuccessor(0, cur_action)
                    score = max(score, recursiveAlphaBeta(next_state, depth, next_agent, alpha, beta))
                    if score > beta:
                        break
                    alpha = max(alpha, score)
            else:
                score = float('inf')
                for cur_action in legal_actions:
                    next_state = state.generateSuccessor(agent, cur_action)
                    if next_agent == 0:
                        score = min(score, recursiveAlphaBeta(next_state, depth - 1, next_agent, alpha, beta))
                    else:
                        score = min(score, recursiveAlphaBeta(next_state, depth, next_agent, alpha, beta))
                    if score < alpha:
                        break
                    beta = min(beta, score)
            return score

        return alphaBeta(gameState, self.depth)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax(state, depth):
            if (state.isWin() or state.isLose()) or depth == 0:
                return None

            legal_actions = state.getLegalActions(0)

            action = None
            score = float('-inf')
            for cur_action in legal_actions:
                next_state = state.generateSuccessor(0, cur_action)
                cur_score = recursiveExpectimax(next_state, depth, 1)
                if cur_score > score:
                    score = cur_score
                    action = cur_action
            return action

        def recursiveExpectimax(state, depth, agent):
            if (state.isWin() or state.isLose()) or depth == 0:
                return self.evaluationFunction(state)

            legal_actions = state.getLegalActions(agent)
            next_agent = (agent + 1) % state.getNumAgents()

            score = 0
            if agent == 0:
                score = float('-inf')
                for cur_action in legal_actions:
                    next_state = state.generateSuccessor(agent, cur_action)
                    score = max(score, recursiveExpectimax(next_state, depth, next_agent))
            else:
                for cur_action in legal_actions:
                    next_state = state.generateSuccessor(agent, cur_action)
                    if next_agent == 0:
                        score += recursiveExpectimax(next_state, depth - 1, next_agent)
                    else:
                        score += recursiveExpectimax(next_state, depth, next_agent)
                score /= len(legal_actions)
            return score

        return expectimax(gameState, self.depth)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # The best & the worst possible states
    if currentGameState.isLose():
        return float("-inf")
    elif currentGameState.isWin():
        return float("inf")

    pacman_pos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()

    # Food parameter
    food_left = currentGameState.getFood().asList()
    food_distances = []
    for food in food_left:
        food_distances.append(util.manhattanDistance(pacman_pos, food))
    nearest_food = min(food_distances)

    # Ghost parameter
    ghosts = currentGameState.getGhostStates()
    active_ghost_distances = []
    for ghost in ghosts:
        if not ghost.scaredTimer:
            active_ghost_distances.append(util.manhattanDistance(pacman_pos, ghost.getPosition()))
    nearest_ghost = 0
    if len(active_ghost_distances) != 0:
        nearest_ghost = min(active_ghost_distances)

    # Coefficients to configure evaluation
    score_coef = 5
    food_left_coef = 4
    nearest_food_coef = 2
    nearest_ghost_coef = 2

    evaluation = score_coef * score - food_left_coef * len(food_left) - nearest_food_coef * nearest_food
    if nearest_ghost != 0:
        evaluation -= nearest_ghost_coef * (1 / nearest_ghost)
    return evaluation


# Abbreviation
better = betterEvaluationFunction
