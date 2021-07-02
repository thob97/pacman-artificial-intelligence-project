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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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

def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    
    def getAction(self, gameState):
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
        #python3 autograder.py -q q2
        #python3 pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4 -> -492
        #python3 pacman.py -p MinimaxAgent -l minimaxClassic -a depth=3 -> 7
        #python3 pacman.py -p MinimaxAgent -l minimaxClassic -a depth=2 -> 8
        #python3 pacman.py -p MinimaxAgent -l minimaxClassic -a depth=1 -> 9
        
        LASTGHOST = gameState.getNumAgents() 
        PACMAN = 0
        FIRSTGHOST = 1

        def max_value(self, state, depth):
            value = float('-inf')
            for action in state.getLegalActions(PACMAN):
                successor = state.generateSuccessor(PACMAN, action)
                value = max(value, min_max(self, successor, depth+1, FIRSTGHOST)) 
            return value

        def min_value(self, state, depth, ghostIndex):
            value = float('inf')
            for action in state.getLegalActions(ghostIndex):
                successor = state.generateSuccessor(ghostIndex,action)
                value = min(value, min_max(self, successor, depth, ghostIndex+1))
            return value

        def min_max(self, state, depth, agentIndex):
            #if isWin or isLose
            #or if depth is reached and all ghosts have made their last move
            if state.isWin() or state.isLose() or (depth >= self.depth and agentIndex==LASTGHOST) :
                return self.evaluationFunction(state)
            #after last ghost made his move, pacman can move again
            # IF MAX
            if agentIndex is LASTGHOST: 
                return max_value(self, state, depth)
            # IF MIN
            else: 
                return min_value(self, state, depth, agentIndex)

        #first max_value iteration. But will return action instead of a value
        def run():
            value = float('-inf')
            bestAction = None
            for action in gameState.getLegalActions(PACMAN):
                successor = gameState.generateSuccessor(PACMAN, action)
                value, bestAction = max( (value, bestAction) , (min_max(self, successor, 1, FIRSTGHOST), action))
            #print(value)
            return bestAction

        return run()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #python3 autograder.py -q q3
        #python3 pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic
        # as fast as...
        #python3 pacman.py -p MinimaxAgent -a depth=3 -l smallClassic

        #python3 pacman.py -p AlphaBetaAgent -l minimaxClassic -a depth=4 -> -492
        #python3 pacman.py -p AlphaBetaAgent -l minimaxClassic -a depth=3 -> 7
        #python3 pacman.py -p AlphaBetaAgent -l minimaxClassic -a depth=2 -> 8
        #python3 pacman.py -p AlphaBetaAgent -l minimaxClassic -a depth=1 -> 9

        LASTGHOST = gameState.getNumAgents() 
        PACMAN = 0
        FIRSTGHOST = 1

        def max_value(self, state, depth, alpha, beta):
            value = float('-inf')
            for action in state.getLegalActions(PACMAN):
                successor = state.generateSuccessor(PACMAN, action)
                value = max(value, min_max(self, successor, depth+1, FIRSTGHOST, alpha, beta)) 
                if value > beta:            #new
                    return value            #
                alpha = max(value, alpha)   #
            return value

        def min_value(self, state, depth, ghostIndex, alpha, beta):
            value = float('inf')
            for action in state.getLegalActions(ghostIndex):
                successor = state.generateSuccessor(ghostIndex,action)
                value = min(value, min_max(self, successor, depth, ghostIndex+1, alpha, beta))
                if value < alpha:       #new
                    return value        #
                beta = min(value, beta) #
            return value

        def min_max(self, state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose() or (depth >= self.depth and agentIndex==LASTGHOST) :
                return self.evaluationFunction(state)
            # IF MAX
            if agentIndex is LASTGHOST: 
                return max_value(self, state, depth, alpha, beta)
            # IF MIN
            else: 
                return min_value(self, state, depth, agentIndex, alpha, beta)
     
        #first max_value iteration. But will return action instead of a value
        def run():
            value = float('-inf')
            alpha = float('-inf')
            beta = float('inf')
            bestAction = None
            for action in gameState.getLegalActions(PACMAN):
                successor = gameState.generateSuccessor(PACMAN, action)
                value, bestAction = max( (value, bestAction) , (min_max(self, successor, 1, FIRSTGHOST, alpha, beta), action))
                if value > beta:            #new
                    return value            #
                alpha = max(value, alpha)   #
            #print(value)
            return bestAction

        return run()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        #python3 autograder.py -q q4
        #python3 pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic
        # as fast as...
        #python3 pacman.py -p MinimaxAgent -a depth=3 -l smallClassic
        #Q3
        #python3 pacman.py -p AlphaBetaAgent -l trappedClassic -a depth=3 -q -n 10
        #python3 pacman.py -p ExpectimaxAgent -l trappedClassic -a depth=3 -q -n 10

        LASTGHOST = gameState.getNumAgents() 
        PACMAN = 0
        FIRSTGHOST = 1

        def max_value(self, state, depth):
            value = float('-inf')
            for action in state.getLegalActions(PACMAN):
                successor = state.generateSuccessor(PACMAN, action)
                value = max(value, min_max(self, successor, depth+1, FIRSTGHOST)) 
            return value

        def exp_value(self, state, depth, ghostIndex):                                  #new
            value = 0                                                                   #
            probability = 1 / len(state.getLegalActions(ghostIndex))                    #
            for action in state.getLegalActions(ghostIndex):                            #
                successor = state.generateSuccessor(ghostIndex,action)                  #
                value += probability * min_max(self, successor, depth, ghostIndex+1)    #
            return value                                                                #

        def min_max(self, state, depth, agentIndex):
            if state.isWin() or state.isLose() or (depth >= self.depth and agentIndex==LASTGHOST) :
                return self.evaluationFunction(state)
            # IF MAX
            if agentIndex is LASTGHOST: 
                return max_value(self, state, depth)
            # IF MIN
            else: 
                return exp_value(self, state, depth, agentIndex)

        def run():
            value = float('-inf')
            bestAction = None
            for action in gameState.getLegalActions(PACMAN):
                successor = gameState.generateSuccessor(PACMAN, action)
                value, bestAction = max( (value, bestAction) , (min_max(self, successor, 1, FIRSTGHOST), action))
            #print(value)
            return bestAction

        return run()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
