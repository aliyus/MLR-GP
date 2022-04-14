# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:23:20 2020

@author: aliyus
"""

TCMODE = 'noTC'

poplnType = 'Features'
FtEvlType = 'MSE'

# ----------------------------------------------------------------------------- 
popsize = 50
runs = 5
nofGen = 5
# ----------------------------------------------------------------------------- 
# settarget=0.04
settarget=""

# Initialise
import csv
import itertools
import operator
import math
import random
import numpy
#import deap
from deap import base
from deap import creator
from deap import tools
from deap import gp
import datetime
import time
#from . import tools
import pandas as pd
import numpy as np
import os
from functools import reduce
from operator import add, itemgetter
from multiprocessing.pool import threading #ThreadPool, threading
# from deap_p5 import tctools
#from multiprocessing.pool import ThreadPool, threading


# Usetime tags for output files
run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M") #           (2)
#===============================================================================

def div(left, right):
    return left / right

# ======= no. of variables will depend on the data ========
#pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 13), float, "x")
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 13), float, "x") #Airfoil
#pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 8), float, "x") #Wine quality
#pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 8), float, "x") #Concrete
# ------------------------------ Features OR StdTree -------------------------------FFFFFFFFF
if poplnType == 'Features':                   #                                     FFFFFFFFF
    def feature(left, right):
        pass
    pset.addPrimitive(feature, [float,float], float)
else:
    if poplnType == 'StdTree':
        pass                                 #                                      FFFFFFFFF
# ----------------------------------------------------------------------------------FFFFFFFFF
pset.addPrimitive(operator.add, [float,float], float)
pset.addPrimitive(operator.sub, [float,float], float)
pset.addPrimitive(operator.mul, [float,float], float)
pset.addPrimitive(div, [float,float], float) # ???????????????????
#pset.addPrimitive(math.log, [float], float)
#pset.addPrimitive(math.exp, [float], float)
pset.addPrimitive(operator.neg, [float], float)
pset.addPrimitive(math.cos, [float], float)
pset.addPrimitive(math.sin,[float], float)
pset.addEphemeralConstant("nrand101", lambda: random.randint(1,100)/20, float)  #(3a)
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) #Weight is positive (i.e. maximising problem)  for normalised.
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
"""    
#==============================================================================
#==============================================================================
"""
    
datafolder = "C:\\Users\\Aliyu Sambo\\OneDrive - Birmingham City University\\Experiment_Phase3\\Boston_Housing\\housing.csv"

with open(datafolder) as train:
    trainReader = csv.reader(train)
    Tpoints = list(list(float(item) for item in row) for row in trainReader)
#split data: random selection without replacement
#Tpoints = points.copy()
p = 1 - 0.7 # p = test data
random.seed(2019)   
x1=random.shuffle(Tpoints)   
split = int(len(Tpoints)*p)
datatrain=Tpoints[split:len(Tpoints)]
datatest=Tpoints[0:split]

"""
===============================================================================
# ======== TREE REPRESENTATION (FEATURES or TREES)
===============================================================================
"""
#===============================================================================
## ======== GENERATE FEATURES
#===============================================================================
def poplnFt(popsize):    
    #print(f'Creating a population of individuals made of features.')
    current=[]
    # Function to check validity of ind =======================================
    def checkvalid(ind):
        validity = 'Valid'
        new = ind
        nodes, edges, labels = gp.graph(new)
        if labels[0] != 'feature':
            validity = 'NotValid'
        else :
            for i in range(1,len(labels)): #Range excludes root
                #Check for FUNC nodes
                if labels[i] == 'feature':
                    # Check parent of the FUNC node
                    for r in edges:
                        if r[1] == i:
                            # Mark node as invalid if parent is not FUNC
                            if labels[r[0]] != 'feature':
                                validity = 'NotValid'
        return validity
    while len(current) < popsize:
        # 1. create ind ========================================
        newP = toolbox.population(n=1)
        new = newP[0]
        #Check validity
        if checkvalid(new) == 'NotValid':
            #print('Individual Not Valid')
            pass
        elif  checkvalid(new) == 'Valid' : 
            #print('Valid')
            current.append(new)            
    return current
#------------------------------------------------------------------------------
#==============================================================================
#------------------------------------------------------------------------------
# ======== GENERATE NON FEATURE EXPRESSION
def genNonFt(popsize=1):    
    #print(f'Creating a population of individuals made of features.')
    current=[]
    # Function to check validity of ind =======================================
    def checkvalid(ind):
        validity = 'Valid'
        new = ind
        nodes, edges, labels = gp.graph(new)
        for i in range(len(labels)): #Range excludes root
            #Check for FUNC nodes
            if labels[i] == 'feature':
                validity = 'NotValid'
#                print(labels[i])
        return validity
    while len(current) < popsize:
        newP = toolbox.population(n=1)
        new = newP[0]
        #len(new)
        if checkvalid(new) == 'NotValid':
#            print('non-F Not Valid')
            pass
        else :
#            print('non-F is -- Valid')
            current.append(new)
    return current
#------------------------------------------------------------------------------
#==============================================================================
#------------------------------------------------------------------------------
# Extract Features from an indiviual
def extractfeatures(individual):
    ind = individual
    nodes, edges, labels = gp.graph(ind)
    featuresE = []
    indices = []# indices of FUNC (the parent of a feature).
    for i in range(len(labels)):
        if labels[i] == 'feature':
            indices.append(i)
    # Check child of FUNC - if feature extract     
    for l in indices:
        # Identify the two children of FUNC
        legs = []
        for r in edges:
            if r[0] == l:
                legs.append(r[1])
        # Check both legs of FUNC to get features
        for k in legs:
            # If child is not a feature -> ignore
            if labels[k] == 'feature': pass
            # If child is a feature -> extract
            else:
                slice = ind.searchSubtree(k)
                new1 = gp.PrimitiveTree(ind[:][slice])
                featuresE.append(new1)
        #        print(str(new1))
    return featuresE
#------------------------------------------------------------------------------
"""
===============================================================================
                        EVALUATION FUNCTIONS
===============================================================================
"""
#------------------------------------------------------------------------------
#--------- EVALUATION   v5 (predict and MSE) ---------------------------------- EVALUATION   v5 (predict and MSE)
#------------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def evalfeat(individual, datatrain, datatest):
    time_st = time.perf_counter() # <-- start timing the evaluation
#-------- Extract features from individual ---------------
    featofInd = extractfeatures(individual)
    ftresult = pd.DataFrame()
    # If ZeroDivisionError or ValueError assign worst fitness
    try:                
        for j in range(0,len(featofInd)): 
            func = toolbox.compile(expr=featofInd[j])
            resultlist = []
            # Evaluate Feature j with data
            for item in datatrain:
                Iresult = func(*item[:13])
                resultlist.append(Iresult)
            # Create Column and add result for feature j
            ftresult[f'x{j}']=resultlist
        # Get True Y ----------------------------------------------------------
        Y_actual = []
        for item in datatrain:
            Y_actual.append(item[13])
        # Append True Y -------------------------------------------------------
        ftresult['Y'] = Y_actual
        X = ftresult.loc[:, ftresult.columns != 'Y']
        Y = ftresult['Y']
        #-------------------------------------------
        # Create linear regression object.
        mlr= LinearRegression()
        #------------------------------------------- 
        # Fit linear regression.
        mlr.fit(X, Y)
        # Predict Y - Training Data -----------------------------------
        y_pred = mlr.predict(X) 
        #-------------------------------------------
        MSE = metrics.mean_squared_error(Y_actual, y_pred)
        error = 1/(1+ MSE)          
        # Evaluate Features with TEST data=====================================
        t_ftresult = pd.DataFrame()
        for j in range(0,len(featofInd)): 
            func = toolbox.compile(expr=featofInd[j])
            t_resultlist = []
            for item in datatest:
                Iresult = func(*item[:13])
                t_resultlist.append(Iresult)
            # Create Column and add result for feature j      
            t_ftresult[f'x{j}']=t_resultlist
        # Get Actual Y --------------------------------------------------------
        tY_actual = []
        for item in datatest:
            tY_actual.append(item[13])
        #----------------------------------------------------------------------
        t_ftresult['Y'] = tY_actual
        X = t_ftresult.loc[:, t_ftresult.columns != 'Y']
        Y = t_ftresult['Y']
        #-------------------------------------------
        # Predict
        ty_pred = mlr.predict(X) 
        t_MSE = metrics.mean_squared_error(tY_actual, ty_pred)
        error_test = 1/(1+ t_MSE) 
    except (ZeroDivisionError, ValueError, TypeError):#    except ZeroDivisionError:
        error = 0.010101010101010101
        error_test = 0.010101010101010101  
#        print('zero/value error caught')              
    evln_sec=float((time.perf_counter() - time_st))
#    print('MSE  MSE  MSE  MSE')

    return error, evln_sec, error_test, len(featofInd)

######################################
# GP Mutations                       #
######################################
#def mutUniformFT(individual, expr, pset):
def mutUniformFT(individual):#, pset):
    """Randomly select a point in the tree *individual*, then replace the
    subtree at that point as a root by the expression generated using method
    :func:`expr`.

    :param individual: The tree to be mutated.
    :param expr: A function object that can generate an expression when
                 called.
    :returns: A tuple of one tree.
    """
# individual = v_ind[8]
    sub=[]
    # Choose to mutate with a feature or sub-featue expression
    if random.random() < 0.5:
        #Create a feature
        sub = poplnFt(1)[0]
        subType ='feature'
    else:
        #create a sub-feature
        sub = genNonFt(1)[0]
        subType ='notFUNC'
#    print(f'Subtree Type: {subType}')
    nodes, edges, labels = gp.graph(individual) 
#------------------------------------------------------------------------------ 
    count = 0
    success = False
    while count < 10 and success == False:
        count += 1
        index = random.randrange(1,len(individual)) # Leave the root node 
        #print(index)
        labels[index]
        edges[index-1][0]
        pointparent = ''
        # Check selected node type 'feature' or 'non-feature'
        if labels[index] == 'feature':
            pointtype = 'feature' 
        else:
            pointtype = 'non-feature'
        #Check point parent type if point not FUNC 
        if pointtype != 'feature':
            #check parenttype
            #print(edges[index-1])
            parent = edges[index-1][0]
            pointparent = labels[parent]
#        print(f'Point Type: {subType}')
        # Substitute if transaction is valid i.e. Type for type OR different and valid
        if (pointtype == subType) or (pointtype != subType and pointparent == 'feature'):
            slice_ = individual.searchSubtree(index)
            individual[slice_] = sub
            success = True
#        checkvalid(individual) # ---------------???????????
    return individual,

""" 
==============================================================================
=======                     OPERATORS                               ==========
==============================================================================
"""
######################################
# GP Crossovers                      #
######################################
def cxOnePointFt(ind1, ind2):
    """Randomly select in each individual and exchange each subtree with the
    point as root between each individual.
    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :returns: A tuple of two trees.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2
    count = 0
    success = False
    while count < 10 and success == False:
        count += 1
        index1 = random.randrange(1,len(ind1)) # Leave the root node 
        index2 = random.randrange(1,len(ind2)) # Leave the root node        
        # CHECK IF SUBTREE EXCHANGE WILL LEAD TO VALID INDIVIDUALS        
        #1.  Check index types ================================================
        # ---- check ind1 ----------------
        nodes, edges, labels = gp.graph(ind1) 
        if labels[index1] == 'feature':
            pointtype1 = 'feature' 
        else:
            pointtype1 = 'non-feature'
        #print(f'pointtype1: {pointtype1}')
        # ---- check parent1
        pointparent1 = ''
        if pointtype1 != 'feature':
            #check parenttype
            #print(edges[index1-1])
            parent1 = edges[index1-1][0]
            pointparent1 = labels[parent1]
            #print(f'Parent: {pointparent1}')
        # ---- check ind2 ----------------
        nodes, edges, labels = gp.graph(ind2) 
        if labels[index2] == 'feature':
            pointtype2 = 'feature' 
        else:
            pointtype2 = 'non-feature'  
        #print(f'pointtype2: {pointtype2}')
        # ---- check parent2 
        pointparent2 = ''
        if pointtype2 != 'feature':
            #check parenttype
            #print(edges[index2-1])
            parent1 = edges[index2-1][0]
            pointparent2 = labels[parent1]  
            #print(f'Parent: {pointparent2}')
        # =====================================================================
        #2.  If condition to produce valid offsprings are met proceed
        if (pointtype1 == pointtype2):
            slice1 = ind1.searchSubtree(index1)
            slice2 = ind2.searchSubtree(index2)
            ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]
            success = True
        # If  FUNC is replacing non-FUNC then ensure that parent of the non-FUNC is a FUNC
        if (pointtype1 != pointtype2):
            if (pointtype1 == 'feature' and pointtype2 == 'non-feature' and pointparent2 == 'feature'):
                slice1 = ind1.searchSubtree(index1)
                slice2 = ind2.searchSubtree(index2)
                ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]
                success = True
            # if the reverse is met - (easy reading)
            elif (pointtype2 == 'feature' and pointtype1 == 'non-feature' and pointparent1 == 'feature'):
                slice1 = ind1.searchSubtree(index1)
                slice2 = ind2.searchSubtree(index2)
                ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]
                success = True
        #print(f'successful crossover = {success}')
        #print(f'attempts = {count}')
    return ind1, ind2

""" 
==============================================================================
====================    INITIALISATION 3  ====================================
==============================================================================
"""

toolbox.register("evaluate", evalfeat)
toolbox.register("mate", cxOnePointFt)
toolbox.register("mutate", mutUniformFT)  
toolbox.register("select", tools.selDoubleTournament)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
# When an over the limit child is generated, it is simply replaced by a randomly selected parent.
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.register("worstfitness", tools.selWorst)
# -----------------------------------------------------------------------------


"""
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#           THE MAIN ALGORITHM THAT EVOLVES (Using Features and Normalised MSE for Fitness)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
def gpDoubleCx(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, run=1, report_csv=None, datatrain=None, datatest=None, target =None):
    """
    This algorithm uses a steadystate approach evolutionary algorithm as popularized 
    by the Darrell Whitley and Joan Kauth’s GENITOR system. The idea is to iteratively 
    breed one offspring, assess its fitness, and then reintroduce it into the population.
    The introduction may mean it replaces preexisting individual.
    """  
    counteval = 0
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]BEGIN
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
	#wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww
    #target = 0.02 							#--------------------------------------------------------------(())
    mettarget = 0 # 0 = not set
	#mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
	#[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]END
	#``````````````````````````````````````````````````````````````````````````````  
    update_lock = threading.Lock()
    counteval_lock = threading.Lock()
    # -------------------------------------------------------------------------
    logbook = tools.Logbook()
    logbook.header = ['run', 'gen', 'nevals'] + (stats.fields if stats else [])
#+++++++++++++++++++++++++++++++++++++++++++++
#Evaluation of Initial Population  (NMSE or AdjR2)
#+++++++++++++++++++++++++++++++++++++++++++++
    for ind in population:
        if poplnType == 'Features' and FtEvlType == 'AdjR2':
            if not ind.fitness.valid:
                xo, yo, zo, noft, MSER2_train, MSER2_test = toolbox.evaluate(ind, datatrain, datatest)
                ind.evlntime = yo,
                ind.testfitness = zo,
                ind.fitness.values = xo,
                ind.nooffeatures = noft, 
                ind.mser2_train = MSER2_train, 
                ind.mser2_test  = MSER2_test,             

                if ind.fitness.values == (0.0101010101010101,) :
                    ind.fitness.values = 0.0, #for maximising
                if ind.testfitness == (0.0101010101010101,) :
                    ind.testfitness = 0.0, #for maximising         

        elif poplnType == 'Features' and FtEvlType == 'MSE':
                xo, yo, zo, noft = toolbox.evaluate(ind, datatrain, datatest)
                ind.evlntime = yo,
                ind.testfitness = zo,
                ind.fitness.values = xo,
                ind.nooffeatures = noft,         

                if ind.fitness.values == (0.0101010101010101,) :
                    ind.fitness.values = 0.0, #for maximising
                if ind.testfitness == (0.0101010101010101,) :
                    ind.testfitness = 0.0, #for maximising   
    #+++++++++++++++++++++++++++++++++++++++++++++
    try:
        halloffame.update(population)
    except AttributeError:
        pass
    #+++++++++++++++++++++++++++++++++++++++++++++
    record = stats.compile(population) if stats else {}
    logbook.record(run=run, gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)  
    #+++++++++++++++++++++++++++++++++++++++++++++
    # Capture best individual for Generation 0
    gen=0
    hof_db=[]
    cxOver_db=[] # -------------------77777777777777777777777777777777777777777
    B4Fitness = 0
    B4Test_Fitness = 0
    AfFitness = 0
    AfTest_Fitness = 0
    # -------------------------------------------------------------------------
    # Collect HOF Data  (NMSE or AdjR2) 
    if poplnType == 'Features' and FtEvlType == 'AdjR2':
        hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness), halloffame[0].evlntime, len(halloffame[0]),
                       int(str(halloffame[0].nooffeatures)[1:-2]), float(str(halloffame[0].mser2_train)[1:-2]), float(str(halloffame[0].mser2_test)[1:-2]), str(halloffame[0])])

    elif poplnType == 'Features' and FtEvlType == 'MSE':
        hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness),
                       halloffame[0].evlntime, len(halloffame[0]), int(str(halloffame[0].nooffeatures)[1:-2]), str(halloffame[0])])
    # ------------------------------------------------------------------------- 
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #+++++++++++++ Select for Replacement Function +++++++++++++++
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    from operator import attrgetter
    def selInverseTournament(individuals, k, tournsize, fit_attr="fitness"):
        """Select the worst individual among *tournsize* randomly chosen
        individuals, *k* times. The list returned contains
        references to the input *individuals*.
        
        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param tournsize: The number of individuals participating in each tournament.
        :param fit_attr: The attribute of individuals to use as selection criterion
        :returns: A list of selected individuals.
        
        This function uses the :func:`~random.choice` function from the python base
        :mod:`random` module.
        """
        chosen = []
        for i in range(k):
            aspirants =  [random.choice(individuals) for i in range(tournsize)]
            chosen.append(min(aspirants, key=attrgetter(fit_attr)))
#            for l in aspirants: print(str(l.fitness))
        return chosen
#+++++++++++++++++++++++++++++++++++++++++++++
# Breeding Function - TWO OFFSPRINGS
#+++++++++++++++++++++++++++++++++++++++++++++
    # define a breed function.
    def breed():
        nonlocal population, toolbox, cxpb, mutpb, halloffame, poplnsize, counteval, update_lock, counteval_lock, cxOver_db, mettarget

        # initialise ----------------777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777
        B4Fitness = 0
        B4Test_Fitness = 0
        AfFitness = 0
        AfTest_Fitness = 0
        offspring=[]
        successCX = False
                #++++++++ Select Parents +++++++++++++++++++++++++++++++++++++
        p1, p2 = list(map(toolbox.clone, toolbox.select(population, 2, fitness_size=3, parsimony_size=1.4, fitness_first=False)))
#        p1, p2 = list(map(toolbox.clone, toolbox.select(population, 2)))
        # Fitness Before CrossOver -------------7777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777777
        B4Fitness1 = p1.fitness.values[0]
        B4Test_Fitness1 = p1.testfitness[0]
        
        B4Fitness2 = p2.fitness.values[0]
        B4Test_Fitness2 = p2.testfitness[0]
        #++++++++ Crossover +++++++++++++++++++++++++++++++++++++
        if random.random() < cxpb:
            p1, p2 = toolbox.mate(p1, p2)
            del p1.fitness.values
            del p2.fitness.values
            successCX = True
        #   TWO OFFSPRING  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        offspring = p1, p2
        count=0
        for cand in offspring:  
#            str(cand)
#            count += 1
            #++++++++ mutation on the offspring ++++++++++++++++               
            if random.random() < mutpb:
                cand, = toolbox.mutate(cand)
                del cand.fitness.values
#                print(f'mutated {cand}')
            # Evaluate the offspring if it has changed
            # @@@@@@@@@@@@@@@@@(NMSE or AdjR2)@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            if poplnType == 'Features' and FtEvlType == 'MSE':
                if not cand.fitness.valid:
                    #++++++++ Counting evaluations +++++++++++++++++
#                    counteval_lock.acquire()
                    counteval += 1 #Count the actual evaluations
#                    counteval_lock.release()
                    xo, yo, zo, noft = toolbox.evaluate(cand, datatrain, datatest)
                    cand.evlntime = yo,
                    cand.testfitness = zo,
                    cand.fitness.values = xo, 
                    cand.nooffeatures = noft,
                    # Check if ZeroDivisionError, ValueError 
                    if cand.fitness.values == (0.0101010101010101,) :
                        cand.fitness.values = 0.0, #for maximising
                    if cand.testfitness == (0.0101010101010101,) :
                        cand.testfitness = 0.0, #for maximising  

            elif poplnType == 'Features' and FtEvlType == 'AdjR2':
                if not cand.fitness.valid:
                    #++++++++ Counting evaluations +++++++++++++++++
#                    counteval_lock.acquire()
                    counteval += 1 #Count the actual evaluations
#                    counteval_lock.release()
                    xo, yo, zo, noft, MSER2_train, MSER2_test = toolbox.evaluate(cand, datatrain, datatest)
#                    xo, yo, zo = toolbox.evaluate(p1)                
                    cand.evlntime = yo,
                    cand.testfitness = zo,
                    cand.fitness.values = xo, 
                    cand.nooffeatures = noft,
                    cand.mser2_train = MSER2_train, 
                    cand.mser2_test  = MSER2_test,
                    #Check if ZeroDivisionError, ValueError 
                    if cand.fitness.values == (0.0101010101010101,) :
                        cand.fitness.values = 0.0, #for maximising
                    if cand.testfitness == (0.0101010101010101,) :
                        cand.testfitness = 0.0, #for maximising  
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@            

            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # REPLACEMENT - worst fitness from random k
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#            update_lock.acquire()          # LOCK !!!  
            # Identify a individual to replace from the population. Use Inverse Tournament
            candidates = selInverseTournament(population, k=1, tournsize=5)
            candidate = candidates[0]
            # Replace if offspring is better than candidate individual 
            if cand.fitness.values[0] > candidate.fitness.values[0]: # Max
            # if p1.fitness.values[0] < candidate.fitness.values[0]: # Min
                    population.append(cand) 
                    population.remove(candidate)
#                    print(f'{count} replaced')
#            update_lock.release()            # RELEASE !!!
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # Update hall of fame ---------------------------------------------                                                                    
            try:
                halloffame.update(population)
            except AttributeError:
                pass       
            # Record Fitness Values after Cross-Over
            if successCX == True:
                if count == 1:
                    B4Fitness = B4Fitness1
                    B4Test_Fitness = B4Test_Fitness1
                else:
                    B4Fitness = B4Fitness2
                    B4Test_Fitness = B4Test_Fitness2
                    
                AfFitness = cand.fitness.values[0]
                AfTest_Fitness = cand.testfitness[0]
                
                try:
                    Train_imp = round(100*(AfFitness - B4Fitness)/B4Fitness)
                except ZeroDivisionError:
                    Train_imp = 0
                
                try:
                    Test_imp  = round(100*(AfTest_Fitness - B4Test_Fitness)/B4Test_Fitness)
                except ZeroDivisionError:
                    Test_imp = 0
                    
                cxOver_db.append([run, gen, Train_imp, Test_imp])

#++++++++++++++++++++++++++++++++++++++++++++++++
#  GENERATIONAL STATs Collect Depending on Method
#++++++++++++++++++++++++++++++++++++++++++++++++
    # ----------Crossover - Features OR StdTree ----------------------------------------FFFFFFFFF

    if poplnType == 'Features' and FtEvlType == 'MSE':
    # ----------------------------------------------------------------------------------FFFFFFFFF             
        def collectStatsGen():
            nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv
            #++++++++++ Collect Stats ++++++++++++++++++++
            record = stats.compile(population) if stats else {}
            logbook.record(run= run, gen=gen, nevals=counteval, **record)
            
            if verbose:
                print(logbook.stream) 
            #+++++++++++++++++++++++++++++++++++++++++++++
            #Update hall of fame database for each generation
            hof_db.append([run, gen, str(halloffame[0].fitness), str(halloffame[0].testfitness), 
                           halloffame[0].evlntime, len(halloffame[0]), int(str(halloffame[0].nooffeatures)[1:-2]), str(halloffame[0])])

    

        #+++++++++++++++++++++++++++++++++++++++++++++
    #Create Report for the Run 
        #+++++++++++++++++++++++++++++++++++++++++++++        
    def collectStatsRun():
        nonlocal population, stats, run, gen, counteval, logbook, verbose, hof_db, halloffame, report_csv 

        #Put into dataframe
        chapter_keys = logbook.chapters.keys()
        sub_chaper_keys = [c[0].keys() for c in logbook.chapters.values()]
        
        data = [list(map(itemgetter(*skey), chapter)) for skey, chapter 
                     in zip(sub_chaper_keys, logbook.chapters.values())]
        data = np.array([[*a, *b, *c, *d, *e] for a, b, c, d, e in zip(*data)])
        
        columns = reduce(add, [["_".join([x, y]) for y in s] 
                               for x, s in zip(chapter_keys, sub_chaper_keys)])
        df = pd.DataFrame(data, columns=columns)
        
        keys = logbook[0].keys()
        data = [[d[k] for d in logbook] for k in keys]
        for d, k in zip(data, keys):
            df[k] = d
        #+++++++++++++++++++++++++++++++++++++++++++++
        #Export Report to local file
        if os.path.isfile(report_csv):
            df.to_csv(report_csv, mode='a', header=False)
        else:
            df.to_csv(report_csv)
        #+++++++++++++++++++++++++++++++++++++++++++++
        ## Save 'Hall Of Fame' database
        #++++++++++++++++++++++++++++++++++++++++++++++
        #List to dataframe
        hof_dframe=pd.DataFrame(hof_db, columns=['Run', 'Generation', 'Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'No_of_Features','Best'])
        hof_csv = f'{report_csv[:-4]}_hof.csv'#Destination file (local)
        #Export from dataframe to CSV file. Update if exists
        if os.path.isfile(hof_csv):
            hof_dframe.to_csv(hof_csv, mode='a', header=False)
        else:
            hof_dframe.to_csv(hof_csv)      



#++++++++++++++++++++++++++++++++++++++++++++++++++
#  STATs Crossover Effect
#++++++++++++++++++++++++++++++++++++++++++++++++++  
    def collectcxOverStats():
        nonlocal run, gen, B4Fitness, B4Test_Fitness, AfFitness, AfTest_Fitness, report_csv, cxOver_db     
    #+++++++++++++++++++++++++++++++++++++++++++++
    ## Save Crossover Stats
    #++++++++++++++++++++++++++++++++++++++++++++++
        #List to dataframe
        cxOver_dframe=pd.DataFrame(cxOver_db, columns=['Run', 'Generation', 'Train_Fitness_imp', 'Test_Fitness_imp'])
#        cxOver_dframe=pd.DataFrame(cxOver_db, columns=['Run', 'Generation', 'B4Fitness', 'B4Test_Fitness', 'AfFitness', 'AfTest_Fitness'])
        cxOver_csv = f'{report_csv[:-4]}_cxOver.csv'#Destination file (local)
        #Export from dataframe to CSV file. Update if exists
        if os.path.isfile(cxOver_csv):
            cxOver_dframe.to_csv(cxOver_csv, mode='a', header=False)
        else:
            cxOver_dframe.to_csv(cxOver_csv)
        cxOver_db=[]
		
#+++++++++++++++++++++++++++++++++++++++++++++
# Create a Generation
#+++++++++++++++++++++++++++++++++++++++++++++
    # Begin the generational process
    for gen in range(1, ngen+1):
        # Generate offsprings -  equivalent to a generation / populations size
        poplnsize =  len(population)
#        poplnsize =  500
        counteval = 0 
#        for h in range(poplnsize):
#            breed()
        while counteval < poplnsize:
            breed()
#            for j in range(poplnsize - counteval):
#                breed()
#        collectcxOverStats()
        collectStatsGen()
    collectStatsRun()
###############################################################################       
    return population, logbook    
###############################################################################

#==============================================================================
#==============================================================================
#============ Collect Stats for the Final Generation ==========================
#==============================================================================
#==============================================================================
    

if poplnType == 'Features' and FtEvlType == 'MSE':
#===========================================================
    #Function to collect stats for the last generation
    def lastgenstats(population, toolbox, gen=0,  run=0, report_csv=None, datatrain=None, datatest=None):
    #    nonlocal population, toolbox, report_csv, run, gen
        lastgen_db=[]    
        for j in range(len(population)):
            xo, yo, zo, noft = toolbox.evaluate(population[j], datatrain, datatest)
            population[j].fitness.values = xo,
            population[j].evlntime = yo,
            population[j].testfitness = zo,
            population[j].nooffeatures = noft,
#            population[j].mser2_train = MSER2_train, 
#            population[j].mser2_test  = MSER2_test,    
            lastgen_db.append([run, gen, float(str(population[j].fitness)[1:-2]), float(str(population[j].testfitness)[1:-2]), float(str(population[j].evlntime)[1:-2]),
                               len(population[j]), int(str(population[j].nooffeatures)[1:-2]), str(population[j])])
        lastgen_dframe=pd.DataFrame(lastgen_db, columns=['Run', 'Generation', 'Train_Fitness', 'Test_Fitness', 'Evln_time', 'Length', 'No_of_Features', 'Best'])
        #Destination file
        lastgen_csv = f'{report_csv[:-4]}_lastgen.csv'
        #Export from dataframe to CSV file. Update if exists               
        if os.path.isfile(lastgen_csv):
            lastgen_dframe.to_csv(lastgen_csv, mode='a', header=False)
        else:
            lastgen_dframe.to_csv(lastgen_csv)



"""
============================================================================
Function to create initial population: (1) FIXED SIZE AND (2)  UNIQUE INDIVIDUALS
(Constants are treated as same).
============================================================================
"""
def inipoplnF(popsize):    
#    nonlocal poplnType #-------------------------------------------------------------------??????????????????
    ini_len = 10 # Initial lengths
#    popsize = 500
    print(f'Creating a population of {popsize} individuals - each of size: {ini_len}')
    # Function to extract the node types   ----------------------------------------
    def graph(expr):
        str(expr)
        nodes = range(len(expr))
        edges = list()
        labels = dict()
        stack = []
        for i, node in enumerate(expr):
            if stack:
                edges.append((stack[-1][0], i))
                stack[-1][1] -= 1
            labels[i] = node.name if isinstance(node, gp.Primitive) else node.value
            stack.append([i, node.arity])
            while stack and stack[-1][1] == 0:
                stack.pop()
        return nodes, edges, labels
#    ------------------------------- create 1st individual
    current=[]
    newind=[]
    #-------------------------------------------------------------------??????????????????
#    newind= toolbox.population(n=1)
    if poplnType == 'Features':
        newind = poplnFt(1)# 
    elif poplnType == 'StdTree':
        newind = toolbox.population(1)# 
        #-------------------------------------------------------------------?????            
    while len(newind[0]) != ini_len:
        newind = toolbox.population(n=1)
    current.append(newind[0])
#    ------------------------------- Create others; 
#    For each new one check to see a similar individual exists in the population.
    while len(current) < popsize:
    #-------------------------------------------------------------------??????????????????
    #    newind= toolbox.population(n=1)
        if poplnType == 'Features':
            pop = poplnFt(1)# 
        elif poplnType == 'StdTree':
            pop = toolbox.population(1)# 
            #-------------------------------------------------------------------?????            
        if len(pop[0]) == ini_len:
            # ----------------------------- Check for duplicate
            lnodes, ledges, llabels = graph(pop[0])
            similarity = 'same'
            for k in range(len(current)): # CHECK all INDs in CURRENT population
                nodes, edges, labels = graph(current[k])
                for j in range(len(labels)): # Check NEW against IND from CURRENT
                    constants = 'no' # will use to flag constants
                    if labels[j] != llabels[j]: 
                        similarity = 'different' 
                        # no need to check other nodes as soon as difference is detected 
                    if '.' in str(labels[j]) and '.' in str(llabels[j]): constants = 'yes'
                    if labels[j] != llabels[j] or constants != 'yes': # They are different and not constants
                        continue # no need to check other nodes as soon as difference is detected 
                if similarity =='same': # skips other checks as soon as it finds a match
                    continue
            if similarity == 'different': # add only if different from all existing
                current.append(pop[0])     
    print('population created')
    return current
"""
============================================================================
============================================================================
"""

"""
============================================================================
Function to create initial population: (1) FIXED SIZE AND (2)  UNIQUE INDIVIDUALS
(Constants are treated as same).
============================================================================
"""
def inipopln(popsize):    
    ini_len = 10 # Initial lengths
#    popsize = 500
    print(f'Creating a population of {popsize} individuals - each of size: {ini_len}')
    # Function to extract the node types   ----------------------------------------
    def graph(expr):
        str(expr)
        nodes = range(len(expr))
        edges = list()
        labels = dict()
        stack = []
        for i, node in enumerate(expr):
            if stack:
                edges.append((stack[-1][0], i))
                stack[-1][1] -= 1
            labels[i] = node.name if isinstance(node, gp.Primitive) else node.value
            stack.append([i, node.arity])
            while stack and stack[-1][1] == 0:
                stack.pop()
        return nodes, edges, labels
#    ------------------------------- create 1st individual
    current=[]
    newind=[]
    newind= toolbox.population(n=1)
    while len(newind[0]) != ini_len:
        newind = toolbox.population(n=1)
    current.append(newind[0])
#    ------------------------------- Create others; 
#    For each new one check to see a similar individual exists in the population.
    while len(current) < popsize:
        pop = toolbox.population(n=1)
        if len(pop[0]) == ini_len:
            # ----------------------------- Check for duplicate
            lnodes, ledges, llabels = graph(pop[0])
            similarity = 'same'
            for k in range(len(current)): # CHECK all INDs in CURRENT population
                nodes, edges, labels = graph(current[k])
                for j in range(len(labels)): # Check NEW against IND from CURRENT
                    constants = 'no' # will use to flag constants
                    if labels[j] != llabels[j]: 
                        similarity = 'different' 
                        # no need to check other nodes as soon as difference is detected 
                    if '.' in str(labels[j]) and '.' in str(llabels[j]): constants = 'yes'
                    if labels[j] != llabels[j] or constants != 'yes': # They are different and not constants
                        continue # no need to check other nodes as soon as difference is detected 
                if similarity =='same': # skips other checks as soon as it finds a match
                    continue
            if similarity == 'different': # add only if different from all existing
                current.append(pop[0])     
    print('population created')
    return current
"""
============================================================================
============================================================================
"""

def main():
    random.seed(2019)
# ================================================
    tag = f'{TCMODE}_{poplnType}_{FtEvlType}_BHousing'#Standard GP - Steady State FtEvlType poplnType
# -----------------------------------------------------------------------
        
    reportfolder = f"C:\\Users\\Aliyu Sambo\\OneDrive - Birmingham City University\\Experiment_Ph5\\Features\\TC\\"

    report_csv = f"{reportfolder}PH5_{run_time}_{tag}.csv"
#-----------------------------------------------------------------------
    for i in range(1, runs+1):
        run = i
        pop = poplnFt(popsize)# 
#         #-----------------------------------------
        hof = tools.HallOfFame(1) 
        # Configure Stats accordingly -----------------------------------------               
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        stats_evlntime = tools.Statistics(lambda ind: ind.evlntime)
        stats_testfitness = tools.Statistics(lambda ind: ind.testfitness)
        stats_nooffeatures = tools.Statistics(lambda ind: ind.nooffeatures)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, evlntime=stats_evlntime, testfitness=stats_testfitness, nooffeatures=stats_nooffeatures)
        print('====>>')
        print('Working with FEATURES and Normalised Mean Squared Error (NMSE) for fitness...')
        # ---------------------------------------------------------------------
        mstats.register("avg", numpy.mean)
        mstats.register("std", numpy.std)
        mstats.register("min", numpy.min)
        mstats.register("max", numpy.max)
        # ------------------------ Algorithm and Options ----------------------FFFFFFFFF
        pop, log = gpDoubleCx(pop, toolbox, 0.9, 0.1, nofGen, # 
                                     stats=mstats, halloffame=hof, verbose=True, run=run, report_csv=report_csv, datatrain=datatrain, datatest=datatest)#0.025

        # ---------------------------------------------------------------------FFFFFFFFF  
        # for i in range(0,len(pop)): print(f'{i} - length: {len(pop[i])} - {pop[i]}')
        print('Taking stats for the last generation....')
        #Collect stats for the last generation of each run.
        lastgenstats(pop, toolbox, gen=nofGen, run=run, report_csv=report_csv, datatrain=datatrain, datatest=datatest)#GEN....??  (9b)
#==============================================================================
if __name__ == "__main__":
    main()    




"""    
# #=============== Sample script for drawing trees ================================
# #==============================================================================
# import matplotlib.pyplot as plt
# import networkx as nx
# #pop=population
# for i in range(0,10): 
#     print(f'{i} - length: {len(pop[i])}')
#     plotind(pop[i])
#     str(pop[i])
# """
# # -------------- Plot graphs of an individual and it's features -------------
# def plotind(xind):
#     nodes, edges, labels = gp.graph(xind)
#     str(xind)
#     plt.figure(figsize=(10,7.5))
#     g = nx.Graph(directed=True)
#     g.add_nodes_from(nodes)
#     g.add_edges_from(edges)
#     pos = nx.planar_layout(g)
#     nx.draw_networkx_nodes(g, pos, node_size=600, node_color='#1f78b4')
#     nx.draw_networkx_edges(g, pos)
#     nx.draw_networkx_labels(g, pos, labels, label_pos=10, font_size = 15)
#     plt.show()
#------------------------------------------------------------------------------
