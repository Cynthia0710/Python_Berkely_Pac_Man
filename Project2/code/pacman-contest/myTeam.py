# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
from game import Actions
import copy


def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

class DummyAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        self.height = gameState.data.layout.height
        self.width = gameState.data.layout.width

        self.updateSafeFood(gameState)

        self.edgeList = self.getHomeEdges(gameState)



    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor (Game state object)
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def chooseAction(self, gameState):
        self.locationOfLastEatenFood(gameState)  # detect last eaten food
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        return random.choice(bestActions)

    def nullHeuristic(self, state, problem=None):
        return 0

    def aStarSearch(self, problem, gameState, heuristic=nullHeuristic):

        from util import PriorityQueue
        startState = problem.getStartState()
        priorityQueue = PriorityQueue()
        heur = heuristic(startState, gameState)
        cost = 0
        f = cost + heur
        startNode = (startState, [], cost)
        priorityQueue.push(startNode, f)
        closeList = []
        while not priorityQueue.isEmpty():
            state,path,currentCost= priorityQueue.pop()
            if state not in closeList:
                closeList.append(state)
                if problem.isGoalState(state):
                    return path
                successors = problem.getSuccessors(state)
                for successor in successors:
                    currentPath = list(path)
                    successor_state = successor[0]
                    move = successor[1]
                    cost = successor[2] + currentCost
                    heur = heuristic(successor_state, gameState)
                    if successor_state not in closeList:
                        currentPath.append(move)
                        f = cost + heur
                        successor_node = (successor_state, currentPath, cost)
                        priorityQueue.push(successor_node, f)
        return []

    def PacmanHeuristic(self, myPos, gameState):
        h = 0
        if self.getDistanceToGhost(gameState) <900:
            enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]

            ghosts = [a for a in enemies if not a.isPacman and a.scaredTimer < 2 and a.getPosition() != None]
            if ghosts != None and len(ghosts) > 0:
                minDistanceToGhost =min( [self.getMazeDistance(myPos, ghost.getPosition() )for ghost in ghosts])
                if minDistanceToGhost < 2:
                    h = pow((5 - minDistanceToGhost), 5)
        return h

    leftMoves = 300
    lastLostFoodPostion = (0, 0)
    lastLostFoodEffect = 0  # use to measure the effective of last lost food position
    corners = []
    edgeList = []
    veryDangerFood=[]
    teammateSafefood=[]
    teammateDangerfood=[]

    def NStep(self, gameState, step):
        i = 0
        actions = gameState.getLegalActions(self.index)

        learningFactor = 0.1
        if step > 1:
          values = [
                self.evaluate(gameState, action) + learningFactor * (float)(self.NStep(self.getSuccessor(gameState, action),
                                                                               step - 1)[0]) for action in actions]
          maxValue = max(values)
          bestActions = [a for a, v in zip(actions, values) if v == maxValue]
          return maxValue, bestActions
        elif step == 1:
          values = [self.evaluate(gameState, action) for action in actions]
          maxValue = max(values)
          bestActions = [a for a, v in zip(actions, values) if v == maxValue]
          return maxValue,bestActions

    def monteCarloTree(self, gameState):
        actions = gameState.getLegalActions(self.index)
        values = []
        successors = []
        for action in actions:
            values.append(self.evaluate(gameState, action))
            successors.append(self.getSuccessor(gameState, action))
        sum = 0
        totalValue = 0
        for i in values:
            totalValue += i
        possibility = []
        for i in values:
            try:
                possibility.append(i / totalValue)
            except:
                possibility = values
        # we calculate the possibility to each action firstly

        # now, we start to simulate for each action
        rate = []
        for i in successors:
            rate.append(self.simulate(i, 10))
        i = 0
        print('poss is:', possibility)
        print('rate is', rate)
        print('action', actions)
        while i < len(actions):
            rate[i] = rate[i] * possibility[i]
            i += 1
        maxFlag = 0
        print(rate)
        i = 0
        while i < len(actions):
            if rate[maxFlag] > rate[i]:
                maxFlag = i
            i += 1
        print(actions[maxFlag])
        return actions[maxFlag]

        # this is the function to simulate the work and return the win rate of each state

    def simulate(self, gameState, simulateTimes):
        i = 0
        win = 0
        while i < simulateTimes:
            i += 1
            win += self.randomSimulate(gameState, self.leftMoves)
        return win / simulateTimes

        # this is the function to simulate one state,win return 1, lose return 0, tie return 0.5

    def randomSimulate(self, gameState, moves):

        i = moves
        actions = gameState.getLegalActions(self.index)
        factor = 0.1
        a = random.choice(actions)
        if i > 1:
            values = self.roughEvaluate(gameState, a) + factor * self.randomSimulate(self.getSuccessor(gameState, a),
                                                                                     i - 1)
            return values
        if i == 1:
            return self.roughEvaluate(gameState, a)

    def roughEvaluate(self, gameState, action):
        features = self.getRoughFeatures(gameState, action)
        weights = self.getRoughWeights(gameState, action)
        return features * weights

    def getRoughFeatures(self, gameState, action):
        """
    Returns a counter of features for the state
    """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = 200 - len(self.getFood(gameState).asList())

        return features

    def getRoughWeights(self, gameState, action):
        """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
        return {'successorScore': 1.0}

    def NStep(self, gameState, step):
        i = 0
        actions = gameState.getLegalActions(self.index)

        learningFactor = 0.1
        if step > 1:
            values = [
                self.evaluate(gameState, action) + learningFactor * (float)(
                    self.NStep(self.getSuccessor(gameState, action),
                               step - 1)[0]) for action in actions]
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            return maxValue, bestActions
        elif step == 1:
            values = [self.evaluate(gameState, action) for action in actions]
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            return maxValue, bestActions

    def getVeryDanerFood(self,gameState):
        foodList=self.getFood(gameState).asList()
        self.veryDangerFood=[]
        corners = self.removeCornersBaseOnDistance(gameState, 15)
        for food in foodList:
            if foodList in corners:
                self.veryDangerFood.append(food)
        return self.veryDangerFood

    def updateSafeFood(self, gameState):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if len(invaders) > 0:
            distanceToGhost = self.getDistanceToGhost(gameState)
            self.corners = self.removeCornersBaseOnDistance(gameState, distanceToGhost)
            self.safeFoods = self.getSafeFood(gameState)
            self.dangerFoods = self.getDangerFood(gameState)
        else:
            dist=15-gameState.getAgentState(self.index).numCarrying
            if dist<6:
                dist=6
            self.corners = self.removeCornersBaseOnDistance(gameState, dist)
            self.safeFoods = self.getSafeFood(gameState)
            self.dangerFoods = self.getDangerFood(gameState)
    def updateTeammateSafeFood(self,gameState):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        ours=self.getTeam(gameState)
        teammates=[a for a in ours if a!=self.index]
        teammate=teammates[0]

        if len(invaders) > 0:
            distanceToGhost = min(
                [self.getMazeDistance(gameState.getAgentState(teammate).getPosition(), i.getPosition())
                 for i in invaders])
            self.corners = self.removeCornersBaseOnDistance(gameState, distanceToGhost)
            self.safeFoods = self.getSafeFood(gameState)
            self.dangerFoods = self.getDangerFood(gameState)
            self.safeFoods=[]
            for food in self.dangerFoods:
                self.safeFoods.append(food)
            self.dangerFoods=[]
            foodList=self.getFood(gameState).asList()
            for food in foodList:
                if food not in self.safeFoods:
                    self.dangerFoods.append(food)

        else:
            dist=15-gameState.getAgentState(teammate).numCarrying
            if dist<6:
                dist=6
            self.corners = self.removeCornersBaseOnDistance(gameState, dist)
            self.safeFoods = self.getSafeFood(gameState)
            self.dangerFoods = self.getDangerFood(gameState)
            self.safeFoods = []
            for food in self.dangerFoods:
                self.safeFoods.append(food)
            self.dangerFoods = []
            foodList = self.getFood(gameState).asList()
            for food in foodList:
                if food not in self.safeFoods:
                    self.dangerFoods.append(food)



    def getDistanceToFood(self,gameState):

        foodList=self.getFood(gameState).asList()
        if len(foodList)>0:
            myPos = gameState.getAgentState(self.index).getPosition()
            minDistanceToFood = min([self.getMazeDistance(myPos, edge) for edge in foodList])
            return minDistanceToFood
        else:
            return 999

    def getDistanceToSafeFood(self,gameState):
        if len(self.safeFoods)>0:
            myPos = gameState.getAgentState(self.index).getPosition()
            minDistanceToSafeFood = min([self.getMazeDistance(myPos, edge) for edge in self.safeFoods])
            return minDistanceToSafeFood
        else:
            return 999

    def getDistanceToHome(self, gameState):

        myPos = gameState.getAgentState(self.index).getPosition()
        minDistanceToHome = min([self.getMazeDistance(myPos, edge) for edge in self.edgeList])
        return minDistanceToHome

    def getDistanceToGhost(self, gameState):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]

        if len(invaders) == 0:
            return 999
        if len(invaders) > 0:
            distanceToGhost = min(
                [self.distancer.getDistance(i.getPosition(), gameState.getAgentState(self.index).getPosition())
                 for i in invaders])
            return distanceToGhost

    def getHomeEdges(self, gameState):
        # use to calculate the min distance to home
        edgeList = []
        height = gameState.data.layout.height
        length = gameState.data.layout.width

        if self.red:
            edge = (int)((length - 1) / 2)
            i = 0
            while i < height - 1:
                i += 1
                if not gameState.hasWall(edge, i):
                    edgeList.append((edge, i))
        else:
            edge = (int)((length + 1) / 2)
            i = 0
            while i < height - 1:
                i += 1
                if not gameState.hasWall(edge, i):
                    edgeList.append((edge, i))
        return edgeList

    def getLostFood(self, gameState):
        currentFood = self.getFoodYouAreDefending(self.getCurrentObservation()).asList()
        if (self.getPreviousObservation() is not None):
            previousFood = self.getFoodYouAreDefending(self.getPreviousObservation()).asList()

        else:
            return (0, 0)
        if len(currentFood) < len(previousFood):
            for i in previousFood:

                if i not in currentFood:
                    return i
        else:
            return (0, 0)

    def getSafeFood(self, gameState):
        allcroners = self.corners
        foodList = self.getFood(gameState).asList()
        safeFood = []
        for i in foodList:
            if i not in allcroners:
                safeFood.append(i)
        return safeFood

    def getDangerFood(self, gameState):
        self.dangerFoods = []
        foodList = self.getFood(gameState).asList()
        for food in foodList:
            if food not in self.safeFoods:
                self.dangerFoods.append(food)
        return self.dangerFoods

    def getScareTime(self, gameState):
        myPos = gameState.getAgentState(self.index).getPosition()

        minDistance = 400

        for i in self.getOpponents(gameState):
            if not gameState.getAgentState(i).isPacman and gameState.getAgentState(i).getPosition() != None:
                if self.distancer.getDistance(gameState.getAgentState(i).getPosition(), myPos) < minDistance:
                    minDistance = self.distancer.getDistance(gameState.getAgentState(i).getPosition(), myPos)
                    ghostIndex = i
        try:
            return gameState.getAgentState(ghostIndex).scaredTimer
        except:
            return 0

    def repeatActionDetect(self):
        repeatTimes = 3
        count = 1
        testLength = 4
        breakFlag = False
        j = 1
        while j < 7:
            j += 1
            testLength = j
            goStrightFlag = True

            if len(self.historyAction) == 0 or len(self.historyAction) < (repeatTimes + 1) * testLength:
                return False
            while True:
                i = 0
                while i < testLength:

                    if not self.historyAction[len(self.historyAction) - count * testLength - i - 1] == \
                           self.historyAction[
                               len(self.historyAction) - i - 1]:
                        breakFlag = True
                        break
                    if count > repeatTimes - 1:
                        breakFlag = True
                        break
                    i += 1
                if breakFlag:
                    breakFlag = False
                    break

                count += 1
            k = 0
            while k < j:
                k += 1
                if not self.historyAction[
                           len(self.historyAction) - 1] == self.historyAction[
                           len(self.historyAction) - k - 1]:
                    goStrightFlag = False
            if count > repeatTimes - 1 and not goStrightFlag:
                return True
            else:
                return False

    def getDistanceToCenter(self, gameState, myPos):
        centerList = self.getHomeEdges(gameState)
        height = gameState.data.layout.height
        length = gameState.data.layout.width
        nearest = height
        for location in centerList:
            if abs(location[1] - (height + 1) / 2 < nearest):
                centerLocation = location
                nearest = abs(location[1] - (height + 1) / 2 < nearest)
        return self.getMazeDistance(myPos, centerLocation)

    def getDistanceToTop(self, gameState, myPos):
        centerList = self.getHomeEdges(gameState)
        height = gameState.data.layout.height
        length = gameState.data.layout.width
        nearest = height
        for location in centerList:
            if abs(location[1] - height + 1) < nearest:
                centerLocation = location
                nearest = abs(location[1] - height + 1) < nearest
        return self.getMazeDistance(myPos, centerLocation)

    def distanceToHighEntry(self, gameState, myPos):
        centerList = self.getHomeEdges(gameState)
        height = gameState.data.layout.height
        length = gameState.data.layout.width
        nearest = height
        for location in centerList:
            if abs(location[1] - 3 * (height + 1) / 4 < nearest):
                centerLocation = location
                nearest = abs(location[1] - 3 * (height + 1) / 4 < nearest)
        return self.getMazeDistance(myPos, centerLocation)

    def getHighEntry(self, gameState):

        height = gameState.data.layout.height
        entryPoint = ()
        nearest = height
        myPos = gameState.getAgentState(self.index).getPosition()
        for location in self.edgeList:
            if abs(location[1] - 3 * (height + 1) / 4 < nearest):
                centerLocation = location
                nearest = abs(location[1] - 3 * (height + 1) / 4 < nearest)
                entryPoint = location

        return entryPoint

    def distanceToLowEntry(self, gameState, myPos):
        centerList = self.getHomeEdges(gameState)
        height = gameState.data.layout.height
        length = gameState.data.layout.width
        nearest = height
        for location in centerList:
            if abs(location[1] - 1 * (height + 1) / 4 < nearest):
                centerLocation = location
                nearest = abs(location[1] - 1 * (height + 1) / 4 < nearest)
        return self.getMazeDistance(myPos, centerLocation)

    # This is the function that find the corners in layout
    def removeAllCorners(self, gameState):
        cornerList = []

        myPos = gameState.getAgentState(self.index).getPosition()
        height = gameState.data.layout.height
        length = gameState.data.layout.width

        loopTimes = 15
        while loopTimes > 0:
            loopTimes -= 1
            i = 1
            while i < length - 1:
                j = 1
                while j < height - 1:

                    if gameState.hasWall(i, j) or (i, j) == myPos:
                        j += 1

                        # better function should consider there is no capsule

                        continue
                    else:
                        # this position is surroud by wall in three directionΩ
                        numberOfWalls = 0
                        if gameState.hasWall(i + 1, j) or (i + 1, j) in cornerList:
                            numberOfWalls += 1
                        if gameState.hasWall(i - 1, j) or (i - 1, j) in cornerList:
                            numberOfWalls += 1
                        if gameState.hasWall(i, j + 1) or (i, j + 1) in cornerList:
                            numberOfWalls += 1
                        if gameState.hasWall(i, j - 1) or (i, j - 1) in cornerList:
                            numberOfWalls += 1
                        if numberOfWalls >= 3 and (i, j) not in cornerList:
                            cornerList.append((i, j))
                    j += 1
                i += 1
        return cornerList

    def removeCornersBaseOnDistance(self, gameState, distanceToGhost):
        cornerList = []
        removeCornerList = []
        removeCornerListCopy = []
        capsuleList = self.getCapsules(gameState)
        myPos = gameState.getAgentState(self.index).getPosition()
        height = gameState.data.layout.height
        length = gameState.data.layout.width
        loopTimes = 0
        loopTimes = 1 + (distanceToGhost - 4) / 2

        while loopTimes >= 1:
            loopTimes -= 1
            i = 1

            while i < length - 1:
                j = 1
                while j < height - 1:

                    if gameState.hasWall(i, j) or (i, j) == myPos:
                        j += 1

                        # better function should consider there is no capsule

                        continue
                    else:
                        # this position is surroud by wall in three directionΩ
                        numberOfWalls = 0
                        if gameState.hasWall(i + 1, j) or (i + 1, j) in removeCornerList:
                            numberOfWalls += 1
                        if gameState.hasWall(i - 1, j) or (i - 1, j) in removeCornerList:
                            numberOfWalls += 1
                        if gameState.hasWall(i, j + 1) or (i, j + 1) in removeCornerList:
                            numberOfWalls += 1
                        if gameState.hasWall(i, j - 1) or (i, j - 1) in removeCornerList:
                            numberOfWalls += 1
                        if numberOfWalls >= 3 and (i, j) not in removeCornerList:
                            removeCornerListCopy.append((i, j))
                    j += 1
                i += 1
                for x in removeCornerListCopy:
                    if x not in removeCornerList:
                        removeCornerList.append(x)
        loopTimes = 30

        while loopTimes > 0:
            loopTimes -= 1
            i = 1
            while i < length - 1:
                j = 1
                while j < height - 1:

                    if gameState.hasWall(i, j) or (i, j) == myPos:
                        j += 1

                        # better function should consider there is no capsule

                        continue
                    else:
                        # this position is surroud by wall in three directionΩ
                        numberOfWalls = 0
                        if gameState.hasWall(i + 1, j) or (i + 1, j) in cornerList:
                            numberOfWalls += 1
                        if gameState.hasWall(i - 1, j) or (i - 1, j) in cornerList:
                            numberOfWalls += 1
                        if gameState.hasWall(i, j + 1) or (i, j + 1) in cornerList:
                            numberOfWalls += 1
                        if gameState.hasWall(i, j - 1) or (i, j - 1) in cornerList:
                            numberOfWalls += 1
                        if numberOfWalls >= 3 and (i, j) not in cornerList and (i, j) not in capsuleList:
                            cornerList.append((i, j))
                    j += 1
                i += 1

        for i in removeCornerList:
            try:
                cornerList.remove(i)
            except:
                continue
        cornerDeep = 15
        while cornerDeep > 0:
            cornerDeep -= 1
            for corner in removeCornerList:
                i = corner[0]
                j = corner[1]
                numberOfWalls = 0
                if (i + 1, j) in cornerList:
                    numberOfWalls += 1
                if (i - 1, j) in cornerList:
                    numberOfWalls += 1
                if (i, j + 1) in cornerList:
                    numberOfWalls += 1
                if (i, j - 1) in cornerList:
                    numberOfWalls += 1
                if numberOfWalls >= 1 and (i, j) not in cornerList and (i, j) not in capsuleList:
                    cornerList.append((i, j))
        return cornerList

    def removeCorners(self, gameState):
        myPos = gameState.getAgentState(self.index).getPosition()
        cornerList = []
        height = gameState.data.layout.height
        length = gameState.data.layout.width

        loopTimes = 10
        while loopTimes > 0:
            loopTimes -= 1
            i = 1
            while i < length - 1:
                j = 1
                while j < height - 1:

                    if gameState.hasFood(i, j) or gameState.hasWall(i, j) or (i, j) == myPos:
                        j += 1

                        # better function should consider there is no capsule

                        continue
                    else:
                        # this position is surroud by wall in three direction
                        numberOfWalls = 0
                        if gameState.hasWall(i + 1, j) or (i + 1, j) in cornerList:
                            numberOfWalls += 1
                        if gameState.hasWall(i - 1, j) or (i - 1, j) in cornerList:
                            numberOfWalls += 1
                        if gameState.hasWall(i, j + 1) or (i, j + 1) in cornerList:
                            numberOfWalls += 1
                        if gameState.hasWall(i, j - 1) or (i, j - 1) in cornerList:
                            numberOfWalls += 1
                        if numberOfWalls >= 3 and (i, j) not in cornerList:
                            cornerList.append((i, j))

                    j += 1
                i += 1
        return cornerList

    def stopAction(self):
        features = util.Counter()
        features['stop'] = 100000
        return features

    def justEatFood(self, gameState):
        if len(self.getFood(gameState).asList()) > 2:
            problem = SearchFood(gameState, self, self.index)
            #print("just eat food", self.aStarSearch(problem, self.PacmanHeuristic))
            return self.aStarSearch(problem, gameState, self.PacmanHeuristic)[0]
        else:
            return self.goHome(gameState)

    def eatVeryDanerFood(self,gameState):
        self.getDangerFood(gameState)
        if len(self.getFood(gameState).asList()) > 2 and len(self.veryDangerFood)>0:
            problem = SearchDangerFood(gameState, self, self.index)
            #print("just eat food", self.aStarSearch(problem, self.PacmanHeuristic))
            return self.aStarSearch(problem, gameState, self.PacmanHeuristic)[0]
        else:
            return self.justEatFood(gameState)


    def eatSafeFood(self, gameState):
        problem = SearchSafeFood(gameState, self, self.index)
        if len(self.aStarSearch(problem, self.PacmanHeuristic)) == 0:
            finalAction = "Stop"
        else:
            finalAction = self.aStarSearch(problem, gameState, self.PacmanHeuristic)[0]
        #print("eatsafefood", self.aStarSearch(problem,gameState, self.PacmanHeuristic))
        return finalAction

    def eatCapsule(self, gameState):
        problem = SearchCapsule(gameState, self, self.index)
        if len(self.aStarSearch(problem, self.PacmanHeuristic)) == 0:
            finalAction = "Stop"
        else:
            finalAction = self.aStarSearch(problem, gameState, self.PacmanHeuristic)[0]
        #print("eatcapsule", self.aStarSearch(problem, gameState, self.PacmanHeuristic))
        return finalAction

    def goHome(self, gameState):

        problem = SearchHome(gameState, self, self.index)
        if len(self.aStarSearch(problem, self.PacmanHeuristic)) == 0:
            finalAction = "Stop"
        else:
            finalAction = self.aStarSearch(problem, gameState, self.PacmanHeuristic)[0]
        #print("gohome",self.aStarSearch(problem,gameState,  self.PacmanHeuristic))
        return finalAction

    def escape(self, gameState):
        problem = SearchEscape(gameState, self, self.index)
        if len(self.aStarSearch(problem, self.PacmanHeuristic)) == 0:
            finalAction = "Stop"
        else:
            finalAction = self.aStarSearch(problem, gameState, self.PacmanHeuristic)[0]
        #print("escape", self.aStarSearch(problem,gameState,  self.PacmanHeuristic))
        return finalAction


class OffensiveReflexAgent(DummyAgent):
    escapeEffect = 0
    carryDots = 0  # The number of dots of this agent carried.
    backHomeTimes = 0
    repeatFlag = 0
    isOffensive = True
    historyAction = []
    goOffensive = True

    def stopAction(self):
        features = util.Counter()
        features['stop'] = 100000
        return features

    def justEatFood(self, gameState):
        if len(self.getFood(gameState).asList()) > 2:
            problem = SearchFood(gameState, self, self.index)
            #print("just eat food", self.aStarSearch(problem, gameState,self.PacmanHeuristic))
            return self.aStarSearch(problem, gameState, self.PacmanHeuristic)[0]
        else:
            return self.goHome(gameState)

    def eatSafeFood(self, gameState):
        problem = SearchSafeFood(gameState, self, self.index)
        if len(self.aStarSearch(problem, self.PacmanHeuristic)) == 0:
            finalAction = "Stop"
        else:
            finalAction = self.aStarSearch(problem, gameState, self.PacmanHeuristic)[0]

        #print("eat safe food", self.aStarSearch(problem, gameState,self.PacmanHeuristic),finalAction)
        return finalAction

    def eatCapsule(self, gameState):
        problem = SearchCapsule(gameState, self, self.index)
        if len(self.aStarSearch(problem, self.PacmanHeuristic)) == 0:
            finalAction = "Stop"
        else:
            finalAction = self.aStarSearch(problem, gameState, self.PacmanHeuristic)[0]
        #print("eat capsule", self.aStarSearch(problem, gameState,self.PacmanHeuristic))
        return finalAction

    def goHome(self, gameState):

        problem = SearchHome(gameState, self, self.index)
        if len(self.aStarSearch(problem, self.PacmanHeuristic)) == 0:
            finalAction = self.escape(gameState)
        else:
            finalAction = self.aStarSearch(problem, gameState, self.PacmanHeuristic)[0]
        #print("go home", self.aStarSearch(problem, gameState,self.PacmanHeuristic))
        return finalAction

    def escape(self, gameState):
        problem = SearchEscape(gameState, self, self.index)
        if len(self.aStarSearch(problem, self.PacmanHeuristic)) == 0:
            finalAction = "Stop"
        else:
            finalAction = self.aStarSearch(problem, gameState, self.PacmanHeuristic)[0]
        #print("escape", self.aStarSearch(problem, self.PacmanHeuristic))
        return finalAction

    def recordInformation(self, gameState):
        self.leftMoves -= 1
        
        if self.lastLostFoodEffect > 0:
            self.lastLostFoodEffect -= 1
        if self.escapeEffect > 0:
            self.escapeEffect -= 1

        if not gameState.getAgentState(self.index).isPacman and self.carryDots != 0:
            self.repeatFlag = 0
            self.carryDots = 0
            self.backHomeTimes += 1
            self.escapeEffect = 0

    def recordInformationAfterCurrentStep(self, gameState, finalAction):
        # compute the number of dots that carried.
        successor = self.getFood(self.getSuccessor(gameState, finalAction)).asList()
        currentFoodList = self.getFood(gameState).asList()
        if len(currentFoodList) > len(successor):
            self.carryDots += 1
            self.repeatFlag = 0
            self.escapeEffect = 0

        self.historyAction.append(finalAction)

    def chooseAction(self, gameState):

        start = time.time()

        self.recordInformation(gameState)

        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        ghosts = [a for a in enemies if not a.isPacman]
        invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]

        allScareTime=min([i.scaredTimer for i in enemies])
        minDistanceToHome = self.getDistanceToHome(gameState)
        distanceToGhost = self.getDistanceToGhost(gameState)

        self.updateSafeFood(gameState)

        '''
        for food in self.safeFoods:
          self.debugDraw(food, [100, 100, 255], False)
        for food in self.dangerFoods:
          self.debugDraw(food, [255, 100, 100], False)
        '''


        if len(self.safeFoods) > 0:
            minDistanceToFood = min([self.distancer.getDistance(food, gameState.getAgentState(self.index).getPosition())
                                     for food in self.safeFoods])
        else:
            minDistanceToFood = 99

        scaredTimes = self.getScareTime(gameState)
        if len(self.getFood(gameState).asList()) <= 2:
            finalAction = self.goHome(gameState)
        elif len(ghosts) == 0:
            finalAction = self.justEatFood(gameState)
        elif allScareTime>15:
            finalAction=self.justEatFood(gameState)
        elif scaredTimes > 15:
            # when the oppsite is scared, we just eat food
            finalAction = self.justEatFood(gameState)
        elif scaredTimes > 5:#need to discuss
            # when the oppsite is scared, we just eat food
            finalAction = self.justEatFood(gameState)
            # eat capsule 往后放
        elif len(self.safeFoods) < 1 and len(self.getCapsules(gameState)) != 0 and scaredTimes < 10:
            finalAction = self.eatCapsule(gameState)

        elif len(self.safeFoods) < 1 and len(self.getCapsules(gameState)) == 0 and gameState.getAgentState(
                self.index).numCarrying > 1:
            finalAction = self.goHome(gameState)


        elif gameState.getAgentState(self.index).numCarrying < 1 and (len(self.safeFoods) > 0):
            finalAction = self.eatSafeFood(gameState)

        elif gameState.getAgentState(self.index).numCarrying < 1 and (len(self.safeFoods) == 0):
            finalAction = self.justEatFood(gameState)

        elif gameState.data.timeleft/4 < self.getDistanceToHome(gameState) + 5 \
                or (gameState.getAgentState(
            self.index).numCarrying > 9 + self.backHomeTimes * 8 and self.getDistanceToSafeFood(gameState) > 5):
            finalAction = self.goHome(gameState)

        elif (gameState.getAgentState(
                self.index).numCarrying > 3 and distanceToGhost<6):
            finalAction=self.escape(gameState)


        elif distanceToGhost < 5 and len(self.getCapsules(gameState)) > 0:
            finalAction = self.eatCapsule(gameState)
            # 这里如果ghost在去往capsule的路上会出现问题

        elif len(self.getSafeFood(gameState)) > 0:
            finalAction = self.eatSafeFood(gameState)

        elif len(self.getCapsules(gameState)) > 0:
            finalAction = self.eatCapsule(gameState)

        elif len(self.getCapsules(gameState)) == 0:
            if gameState.getAgentState(
                self.index).numCarrying > 0:
                finalAction = self.goHome(gameState)
            else:
                finalAction = self.justEatFood(gameState)

        elif distanceToGhost < 3:
            finalAction = self.escape(gameState)

        elif self.leftMoves < minDistanceToHome:
            if len(self.getCapsules(gameState)) > 0:
                finalAction = self.eatCapsule(gameState)
            else:
                finalAction = self.goHome(gameState)
                # 这里可以尝试用自杀回家防守优化
        else:
            finalAction = self.justEatFood(gameState)


        self.recordInformationAfterCurrentStep(gameState, finalAction)
        successor=self.getSuccessor(gameState,finalAction)

        for i in invaders:
            if (successor.getAgentState(self.index).getPosition() == i.getPosition() \
                    or successor.getAgentState(self.index).getPosition() == self.start or self.getDistanceToGhost(successor)==1) \
                    and successor.getAgentState(self.index).getPosition() not in self.getCapsules(gameState)\
                    and scaredTimes<2 and gameState.getAgentState(self.index).isPacman:

                actions=actions = gameState.getLegalActions(self.index)
                values = [self.evaluate(gameState, a) for a in actions]
                maxValue = max(values)
                bestActions = [a for a, v in zip(actions, values) if v == maxValue]
                finalAction = random.choice(bestActions)
                #print("death")

        #print("Time", self.index, time.time() - start,finalAction)

        return finalAction

    def getFeatures(self, gameState, action):

        # get basic parameters
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myPos = successor.getAgentState(self.index).getPosition()
        minDistanceToHome=self.getDistanceToHome(gameState)

        # Stop is meaningless, therefore, it should be bad choice


        features['leftCapsules'] = 100 - len(self.getCapsules(successor))

        # Get the corner feature, we assume the corner is nonmeaning, so, avoid them
        if myPos in self.corners:
            features['inCorner'] = 1
        features['distanceToFood']=self.getDistanceToSafeFood(successor)
        # Get the feature of distance to ghost, once observe the ghost, and distance<5, return to home
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if action == 'Stop':
            try:
                x = min([self.distancer.getDistance(i.getPosition(), gameState.getAgentState(self.index).getPosition())
                         for i in invaders])
                if x != 2:
                    return self.stopAction()
            except:
                return self.stopAction()
        # 获取吃超级豆子之后 Ghost害怕还剩余的时间
        scaredTimes = self.getScareTime(successor)
        if scaredTimes > 3:
            # when the oppsite is scared, we just eat food
            features['inCorner'] = 0
            features['distanceToGhost'] = 0
            features['distanceToHome'] = 0
        elif scaredTimes <= 2:
            # when the oppsite is not scared
            if len(invaders) > 0:
                successroDistanceToGhost = min([self.distancer.getDistance(i.getPosition(), myPos)
                                                for i in invaders])
                distanceToGhost = min(
                    [self.distancer.getDistance(i.getPosition(), gameState.getAgentState(self.index).getPosition())
                     for i in invaders])
                for i in invaders:
                    if myPos == i.getPosition() or myPos ==self.start:
                        features['meetGhost'] = 1
                if distanceToGhost < 5:
                    try:
                        distanceToCapsule = 5 * min(
                            [self.distancer.getDistance(i, myPos) for i in self.getCapsules(successor)])
                        features['distanceToCapsule'] = distanceToCapsule
                    except:
                        distanceToCapsule = -1
            if distanceToGhost < 8:
                try:
                    distanceToCapsule = min([self.distancer.getDistance(i, myPos) for i in self.getCapsules(successor)])
                    features['distanceToCapsule'] = (distanceToCapsule) * 80
                except:
                    features['distanceToHome'] = - minDistanceToHome
            if distanceToGhost < 6:
                features['distanceToGhost'] = 100 - successroDistanceToGhost
                features['successorScore'] = 0

        if self.leftMoves < minDistanceToHome + 4 and self.carryDots > 0:
            # should go home directly
            features['distanceToHome'] = - minDistanceToHome
            features['distanceToFood'] = 0
        elif self.leftMoves < minDistanceToHome:
            try:
                distanceToCapsule = min([self.distancer.getDistance(i, myPos) for i in self.getCapsules(successor)])

                features['distanceToCapsule'] = (distanceToCapsule) * 80
            except:
                distanceToCapsule = -1

        #print("offensive feature:",action,features)
        return features

    def getWeights(self, gameState, action):
        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            return {'leftCapsules': 100,  # Eat capsule
                    'distanceToGhost': -100, 'finalDistanceToHome': 5, 'distanceToCapsule': -1,
                    # distance attribute when come back home
                    'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': 100, 'distanceToLostFood': -5,
                    'distanceToCenter': -3,  # defensive attribute
                    'stop': -10, 'inCorner': -100000, 'reverse': -2,'meetGhost': -1000000,}
        return {'successorScore': 1000, 'leftCapsules': 200,  # Eat food or capsule when it can
                'distanceToGhost': -100, 'distanceToHome': 60, 'distanceToFood': -2, 'distanceToCapsule': -1,
                'distanceToEntry': -20,  # distance attribute when it is pacman
                'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': 100, 'distanceToLostFood': -5,
                'distanceToCenter': -3,  # defensive attribute
                'inCorner': -100000, 'stop': -10, 'reverse': -2, 'meetGhost': -1000000,
                'changeEntryPoint': 1000}


class DefensiveReflexAgent(DummyAgent):
    escapeEffect = 0
    carryDots = 0  # The number of dots of this agent carried.
    backHomeTimes = 0
    repeatFlag = 0
    isOffensive = True
    historyAction = []
    goOffensive = True

    def recordInformation(self, gameState):
        self.leftMoves -= 1
        if self.lastLostFoodEffect > 0:
            self.lastLostFoodEffect -= 1
        if self.escapeEffect > 0:
            self.escapeEffect -= 1

        if not gameState.getAgentState(self.index).isPacman and self.carryDots != 0:
            self.repeatFlag = 0
            self.carryDots = 0
            self.backHomeTimes += 1
            self.escapeEffect = 0

        lostFoodPosition = self.getLostFood(gameState)

        if lostFoodPosition != (0, 0):
            self.lastLostFoodPostion = lostFoodPosition
            self.lastLostFoodEffect = 20

    def recordInformationAfterCurrentStep(self, gameState, finalAction):
        # compute the number of dots that carried.
        successor = self.getFood(self.getSuccessor(gameState, finalAction)).asList()
        currentFoodList = self.getFood(gameState).asList()
        if len(currentFoodList) > len(successor):
            self.carryDots += 1
            self.repeatFlag = 0
            self.escapeEffect = 0

        self.historyAction.append(finalAction)

    def chooseAction(self, gameState):
        start = time.time()
        self.recordInformation(gameState)

        actions = gameState.getLegalActions(self.index)

        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.isPacman]
        self.updateSafeFood(gameState)
        #self.updateTeammateSafeFood(gameState)
        #print(self.safeFoods)
        #print(self.dangerFoods)
        '''
        for food in self.safeFoods:
            self.debugDraw(food, [100, 100, 255], False)
        for food in self.dangerFoods:
            self.debugDraw(food, [255, 100, 100], False)
        '''

        # if number of invaders is less than two, we can go out and try to eat some food

        # when number of invader > 0, we excute defendense strategy
        if gameState.data.timeleft > 1050:

            values = [self.evaluate(gameState, a) for a in actions]
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            finalAction = random.choice(bestActions)
        elif gameState.getAgentState(self.index).scaredTimer > self.getDistanceToHome(gameState) or len(invaders) < 1:

            ghosts = [a for a in enemies if not a.isPacman]
            enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
            invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]

            minDistanceToHome = self.getDistanceToHome(gameState)
            distanceToGhost = self.getDistanceToGhost(gameState)
            #if we need to create a new safefood attribute
            self.updateSafeFood(gameState)
            if len(self.safeFoods) > 0:
                minDistanceToFood = min(
                    [self.distancer.getDistance(food, gameState.getAgentState(self.index).getPosition())
                     for food in self.safeFoods])
            else:
                minDistanceToFood = 99
            scaredTimes = self.getScareTime(gameState)

            if len(self.getFood(gameState).asList()) <= 2:
                finalAction = self.goHome(gameState)
            elif scaredTimes > 15:
                # when the oppsite is scared, we just eat food
                finalAction = self.eatVeryDanerFood(gameState)
            elif scaredTimes > 5:  # need to discuss
                # when the oppsite is scared, we just eat food
                finalAction = self.justEatFood(gameState)

            elif len(self.safeFoods) < 1 and len(self.getCapsules(gameState)) != 0 and scaredTimes < 10:
                finalAction = self.eatCapsule(gameState)

            elif len(self.safeFoods) < 1 and len(self.getCapsules(gameState)) == 0 and gameState.getAgentState(
                    self.index).numCarrying > 1:
                finalAction = self.goHome(gameState)

            elif gameState.getAgentState(self.index).numCarrying < 1 and (len(self.safeFoods) > 0):
                finalAction = self.eatSafeFood(gameState)

            elif gameState.getAgentState(self.index).numCarrying < 1 and (len(self.safeFoods) == 0):
                finalAction = self.justEatFood(gameState)

            elif  gameState.data.timeleft/4 < self.getDistanceToHome(gameState) + 5 \
                    or (gameState.getAgentState(
                self.index).numCarrying > 9 + self.backHomeTimes * 8 and minDistanceToFood > 5):
                finalAction = self.goHome(gameState)

            elif distanceToGhost < 7 and len(self.getCapsules(gameState)) > 0:
                finalAction = self.eatCapsule(gameState)
                # 这里如果ghost在去往capsule的路上会出现问题
            elif distanceToGhost<7 and len(self.getCapsules(gameState)) == 0 and gameState.getAgentState(self.index).numCarrying > 1:
                finalAction =self.escape(gameState)
            elif len(self.getSafeFood(gameState)) > 0:
                finalAction = self.eatSafeFood(gameState)
            elif len(self.getCapsules(gameState)) > 0:
                finalAction = self.eatCapsule(gameState)
            elif len(self.getCapsules(gameState)) == 0:
                if self.carryDots > 0:
                    finalAction = self.goHome(gameState)
                else:
                    finalAction = self.justEatFood(gameState)
            elif distanceToGhost < 3:
                finalAction = self.escape(gameState)

            elif self.leftMoves < minDistanceToHome:
                if len(self.getCapsules(gameState)) > 0:
                    finalAction = self.eatCapsule(gameState)
                else:
                    finalAction = self.goHome(gameState)
                    # 这里可以尝试用自杀回家防守优化

            else:
                finalAction = self.goHome(gameState)

            successor = self.getSuccessor(gameState, finalAction)
            for i in invaders:

                if (successor.getAgentState(self.index).getPosition() == i.getPosition() \
                    or successor.getAgentState(self.index).getPosition() == self.start or self.getDistanceToGhost(
                            successor) == 1) \
                        and successor.getAgentState(self.index).getPosition() not in self.getCapsules(gameState) \
                        and self.getScareTime(gameState) < 2 and gameState.getAgentState(self.index).isPacman\
                        and gameState.getAgentState(self.index).numCarrying > 2:
                    actions = actions = gameState.getLegalActions(self.index)
                    # print(self.getDistanceToGhost(successor)==1)
                    values = [self.evaluateEscape(gameState, a) for a in actions]
                    maxValue = max(values)
                    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
                    finalAction = random.choice(bestActions)



        elif gameState.getAgentState(self.index).isPacman and self.getDistanceToHome(gameState)>1:
            finalAction = self.goHome(gameState)
            successor = self.getSuccessor(gameState, finalAction)
            enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
            invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
            for i in invaders:
                if (successor.getAgentState(self.index).getPosition() == i.getPosition() \
                    or successor.getAgentState(self.index).getPosition() == self.start or self.getDistanceToGhost(
                            successor) == 1) \
                        and successor.getAgentState(self.index).getPosition() not in self.getCapsules(gameState) \
                        and self.getScareTime(gameState) < 2 and gameState.getAgentState(self.index).isPacman \
                        and gameState.getAgentState(self.index).numCarrying > 2:
                    actions = actions = gameState.getLegalActions(self.index)
                    # print(self.getDistanceToGhost(successor)==1)
                    values = [self.evaluateEscape(gameState, a) for a in actions]
                    maxValue = max(values)
                    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
                    finalAction = random.choice(bestActions)

        else:

            values = [self.evaluate(gameState, a) for a in actions]
            maxValue = max(values)
            bestActions = [a for a, v in zip(actions, values) if v == maxValue]
            finalAction = random.choice(bestActions)



        #print("Time",gameState.data.timeleft, self.index, time.time() - start,finalAction)

        self.recordInformationAfterCurrentStep(gameState, finalAction)
        return finalAction

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)


        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        if not successor.getAgentState(self.index).isPacman:
            features['onDefense']=1

        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]

        # before find the invaders, we use  lost food position to estimate the position of invaders
        if len(invaders) == 0:
            if self.lastLostFoodPostion != (0, 0) and self.lastLostFoodEffect > 0:
                features['distanceToLostFood'] = self.getMazeDistance(myPos, self.lastLostFoodPostion)
            elif self.lastLostFoodEffect == 0:
                features['distanceToCenter'] = self.getDistanceToCenter(gameState, myPos)
        #print("defensive features",action,features)
        return features

    def getWeights(self, gameState, action):
        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            return {'leftCapsules': 100,  # Eat capsule
                    'distanceToGhost': -100, 'finalDistanceToHome': 5, 'distanceToCapsule': -1,
                    # distance attribute when come back home
                    'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -100, 'distanceToLostFood': -5,
                    'distanceToCenter': -3,  # defensive attribute
                    'stop': -10, 'inCorner': -100000, 'reverse': -2,'meetGhost': -1000000,}
        return { 'leftCapsules': 200,  # Eat food or capsule when it can
                'distanceToGhost': -100, 'distanceToHome': 60, 'distanceToFood': -2, 'distanceToCapsule': -1,
                'distanceToEntry': -20,  # distance attribute when it is pacman
                'numInvaders': -100000, 'onDefense': 5, 'invaderDistance': -100, 'distanceToLostFood': -20,
                'distanceToCenter': -10,  # defensive attribute
                'inCorner': -100000, 'stop': -10, 'reverse': -2, 'meetGhost': -1000000,}

    def getEscapeFeatures(self, gameState, action):

        # get basic parameters
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myPos = successor.getAgentState(self.index).getPosition()
        minDistanceToHome = self.getDistanceToHome(gameState)

        # Stop is meaningless, therefore, it should be bad choice

        features['leftCapsules'] = 100 - len(self.getCapsules(successor))

        # Get the corner feature, we assume the corner is nonmeaning, so, avoid them
        if myPos in self.corners:
            features['inCorner'] = 1
        features['distanceToFood'] = self.getDistanceToSafeFood(successor)
        # Get the feature of distance to ghost, once observe the ghost, and distance<5, return to home
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if action == 'Stop':
            try:
                x = min([self.distancer.getDistance(i.getPosition(), gameState.getAgentState(self.index).getPosition())
                         for i in invaders])
                if x != 2:
                    return self.stopAction()
            except:
                return self.stopAction()
        # 获取吃超级豆子之后 Ghost害怕还剩余的时间
        scaredTimes = self.getScareTime(successor)
        if scaredTimes > 3:
            # when the oppsite is scared, we just eat food
            features['inCorner'] = 0
            features['distanceToGhost'] = 0
            features['distanceToHome'] = 0
        elif scaredTimes <= 2:
            # when the oppsite is not scared
            if len(invaders) > 0:
                successroDistanceToGhost = min([self.distancer.getDistance(i.getPosition(), myPos)
                                                for i in invaders])
                distanceToGhost = min(
                    [self.distancer.getDistance(i.getPosition(), gameState.getAgentState(self.index).getPosition())
                     for i in invaders])
                for i in invaders:
                    if myPos == i.getPosition() or myPos == self.start:
                        features['meetGhost'] = 1
                if distanceToGhost < 5:
                    try:
                        distanceToCapsule = 5 * min(
                            [self.distancer.getDistance(i, myPos) for i in self.getCapsules(successor)])
                        features['distanceToCapsule'] = distanceToCapsule
                    except:
                        distanceToCapsule = -1
            if distanceToGhost < 8:
                try:
                    distanceToCapsule = min([self.distancer.getDistance(i, myPos) for i in self.getCapsules(successor)])
                    features['distanceToCapsule'] = (distanceToCapsule) * 80
                except:
                    features['distanceToHome'] = - minDistanceToHome
            if distanceToGhost < 6:
                features['distanceToGhost'] = 100 - successroDistanceToGhost
                features['successorScore'] = 0

        if self.leftMoves < minDistanceToHome + 4 and self.carryDots > 0:
            # should go home directly
            features['distanceToHome'] = - minDistanceToHome
            features['distanceToFood'] = 0
        elif self.leftMoves < minDistanceToHome:
            try:
                distanceToCapsule = min([self.distancer.getDistance(i, myPos) for i in self.getCapsules(successor)])

                features['distanceToCapsule'] = (distanceToCapsule) * 80
            except:
                distanceToCapsule = -1

        #print("deffensive escape feature:",action,features)
        return features


    def evaluateEscape(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getEscapeFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights



class PositionSearchProblem:
    """
    It is the ancestor class for all the search problem class.
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point.
    """

    def __init__(self, gameState, agent, agentIndex=0, costFn=lambda x: 1):
        self.walls = gameState.getWalls()
        self.costFn = costFn
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):

        util.raiseNotDefined()

    def getSuccessors(self, state):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append((nextState, action, cost))

        # Bookkeeping for display purposes
        self._expanded += 1  # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        if actions == None: return 999999
        x, y = self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x, y))
        return cost


class SearchFood(PositionSearchProblem):
    """
     The goal state is to find all the food
    """

    def __init__(self, gameState, agent, agentIndex=0):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = agent.getFood(gameState)
        self.capsule = agent.getCapsules(gameState)
        # Store info for the PositionSearchProblem (no need to change this)
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE
        self.carry = gameState.getAgentState(agentIndex).numCarrying
        self.foodLeft = len(self.food.asList())

    def isGoalState(self, state):
        return state in self.food.asList()


class SearchFoodNotInCorners(PositionSearchProblem):
    """
       The goal state is to find all the food
    """

    def __init__(self, gameState, agent, agentIndex=0):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = agent.getFood(gameState)
        self.capsule = agent.getCapsules(gameState)
        # Store info for the PositionSearchProblem (no need to change this)
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE
        self.carry = gameState.getAgentState(agentIndex).numCarrying
        self.foodLeft = len(self.food.asList())
        self.foodNotInCorners = agent.getSafeFood(gameState)

    def isGoalState(self, state):
        return state in self.foodNotInCorners


class SearchSafeFood(PositionSearchProblem):
    """
    The goal state is to find all the safe fooof
    """

    def __init__(self, gameState, agent, agentIndex=0):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = agent.getFood(gameState)
        self.capsule = agent.getCapsules(gameState)
        # Store info for the PositionSearchProblem (no need to change this)
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE
        self.carry = gameState.getAgentState(agentIndex).numCarrying
        self.foodLeft = len(self.food.asList())
        self.safeFood = agent.safeFoods

    def isGoalState(self, state):
        return state in self.safeFood
class SearchDangerFood(PositionSearchProblem):
    """
    The goal state is to find all the safe fooof
    """

    def __init__(self, gameState, agent, agentIndex=0):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = agent.getFood(gameState)
        self.capsule = agent.getCapsules(gameState)
        # Store info for the PositionSearchProblem (no need to change this)
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE
        self.carry = gameState.getAgentState(agentIndex).numCarrying
        self.foodLeft = len(self.food.asList())
        self.dangerFood = agent.veryDangerFood

    def isGoalState(self, state):
        return state in self.dangerFood

class SearchDangerousFood(PositionSearchProblem):
    """
    Used to get the safe food
    """

    def __init__(self, gameState, agent, agentIndex=0):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = agent.getFood(gameState)
        self.capsule = agent.getCapsules(gameState)
        # Store info for the PositionSearchProblem (no need to change this)
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE
        self.carry = gameState.getAgentState(agentIndex).numCarrying
        self.foodLeft = len(self.food.asList())
        self.dangerousFood = agent.dangerFoods

    def isGoalState(self, state):
        return state in self.dangerousFood


class SearchEscape(PositionSearchProblem):
    """
    Used to escape
    """

    def __init__(self, gameState, agent, agentIndex=0):
        "Stores information from the gameState.  You don't need to change this."

        self.food = agent.getFood(gameState)
        self.capsule = agent.getCapsules(gameState)
        # Store info for the PositionSearchProblem (no need to change this)
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE
        self.edgeList = agent.edgeList
        self.safeFood = agent.safeFoods

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        return state in self.edgeList or state in self.capsule


class SearchEntry(PositionSearchProblem):

    def __init__(self, gameState, agent, agentIndex=0):
        self.food = agent.getFood(gameState)
        self.capsule = agent.getCapsules(gameState)
        # Store info for the PositionSearchProblem (no need to change this)
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE
        self.entry = agent.getHighEntry(gameState)

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        return state in self.entry


class SearchHome(PositionSearchProblem):
    def __init__(self, gameState, agent, agentIndex=0):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = agent.getFood(gameState)
        self.capsule = agent.getCapsules(gameState)
        # Store info for the PositionSearchProblem (no need to change this)
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE
        self.edgeList = agent.edgeList

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):

        return state in self.edgeList


class SearchCapsule(PositionSearchProblem):
    def __init__(self, gameState, agent, agentIndex=0):

        self.food = agent.getFood(gameState)
        self.capsule = agent.getCapsules(gameState)
        # Store info for the PositionSearchProblem (no need to change this)
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        self.walls = gameState.getWalls()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def isGoalState(self, state):

        return state in self.capsule


