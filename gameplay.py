from __future__ import division
from math import sqrt
import random as rnd
import numpy as np


# 0 "Scissors"
# 1 "Rock"
# 2 "Paper"
# 3 "Spock"
# 4 "Reptile"

class Game:
    def __init__(self):

        self.NoOfSigns = 5
        self.Names = list(("Scissors","Rock","Paper","Spock","Reptile"))
        self.RandNames = list(("Pipe","Dragon","Meteor","MachineGun","Wand","Bird","Unicorn","Shield","Sword","Castle","Goblin","Staff","Goat"))

        self.Table = np.array(( ( 0, -1,  1, -1 , 1),
                                ( 1,  0, -1, -1 , 1),
                                (-1,  1,  0,  1 ,-1),
                                ( 1,  1, -1,  0 ,-1),
                                (-1, -1,  1, -1 , 0),))
        
        self.SignOrderCount = self.NoOfSigns*np.ones((self.NoOfSigns,self.NoOfSigns,self.NoOfSigns))
        self.LastOrder = np.array((0,0,0))

        self.wins, self.ties, self.losses = 0,0,0
        self.SignCount = 0


    def CheckResult(self,a, b):
        return self.Table[a,b]

    
    def Expand(self):

        
        self.Table = np.vstack(( np.hstack(( self.Table,np.zeros((self.NoOfSigns,1)) )),np.zeros((1,self.NoOfSigns+1)) ))
        A=np.arange(self.NoOfSigns)
        np.random.shuffle(A)
        for i in range(self.NoOfSigns):
            if i<self.NoOfSigns/2:
                self.Table[A[i],self.NoOfSigns] = 1
            else:
                self.Table[A[i],self.NoOfSigns] = -1

            self.Table[self.NoOfSigns,A[i]] = - self.Table[A[i],self.NoOfSigns] 
        print(self.Table)
        self.NoOfSigns = self.NoOfSigns+1
        SignOrderCountOld = self.SignOrderCount
        self.SignOrderCount = self.NoOfSigns*np.ones((self.NoOfSigns,self.NoOfSigns,self.NoOfSigns))
        self.SignOrderCount[0:self.NoOfSigns-1,0:self.NoOfSigns-1,0:self.NoOfSigns-1] = SignOrderCountOld
        self.Names.append(rnd.choice(self.RandNames))
        self.RandNames.remove(self.Names[-1])
        print(self.Names)
        return True


    def PlayerResponse(self,x):
        
        if(self.SignCount < 2):
            answer =  rnd.randint(0,4) 
            self.LastOrder[0:2]=self.LastOrder[1:]
            self.LastOrder[2]=x
            self.SignOrderCount[:,:,self.LastOrder[2]]+=0.2
        else:
            self.LastOrder[0:2]=self.LastOrder[1:]
            count = np.empty(self.NoOfSigns)
            for i in range(len(count)):
                count[i] = self.SignOrderCount[self.LastOrder[0],self.LastOrder[1],i]
            tot_count = np.sum(count)

            q_dist = np.zeros(self.NoOfSigns)
            for i in range(len(q_dist)-1):
                q_dist[i] =  count[i]/tot_count 
            q_dist[-1]=1-np.sum(q_dist)
            
            result = np.zeros(self.NoOfSigns)
            for i in range(len(result)):
                result[i] = max(np.matmul(q_dist,self.Table[i,:].T),0)
            
            resultnorm = np.sum(result)#np.linalg.norm(result)
            for i in range(len(result)-1):
                result[i] = result[i]/resultnorm
            result[-1]=1-np.sum(result)
            
            y = rnd.uniform(0,1)

            found = False
            i = -1
            while not(found):
                i += 1
                if y <= np.sum(result[0:i+1]):
                    answer = i
                    found = True
                
                

            #update dictionary
            self.LastOrder[2]=x
            self.SignOrderCount[self.LastOrder[0],self.LastOrder[1],self.LastOrder[2]]+=1
            #print(result)

        self.SignCount += 1

        #self.last2 = self.last2[1] + x
        
        print(x)
        print(answer)
        print( "You played: " + self.Names[x] + " \nI played:  " + self.Names[answer] + " \nGAME RESULT:" + str(self.CheckResult(x,answer)) )

        if self.CheckResult(x,answer) == -1:
            self.losses += 1
        elif self.CheckResult(x,answer) == 0:
            self.ties   += 1
        elif self.CheckResult(x,answer) == 1:
            self.wins   += 1

        print( " Wins: " + str(self.wins) + " Losses: " + str(self.losses) + " Ties: " + str(self.ties) )
