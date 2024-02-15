#!pip install pyomo
import pyomo.environ as pe
import os
import random
import pandas as pd
from math import sqrt
import math
import matplotlib.pyplot as plt
os.environ['NEOS_EMAIL'] = 'XXXXXX@gmail.com'
global p,p1,p2,working_hours,M,c,k,Np,Nc,df,c1,c2

model = pe.ConcreteModel()

Np = 14
Nc = 4
patients = [p for p in range(1,Np+1)]
points = [0]+patients
print(points)
caregivers = [f"C{c}" for c in range(1,Nc+1)]
skill_level = {c:random.randint(1,5) for c in caregivers }



# df = pd.DataFrame()
# df['X'] = [random.random() for _ in range(Np+1)]
# df['y'] = [random.random() for _ in range(Np+1)]
# df.loc[0,'X']=0.5
# df.loc[0,'y']=0.5
data = {'X': [random.random() for _ in range(Np + 1)],
        'y': [random.random() for _ in range(Np + 1)]}
data['X'][0] = 0.5
data['y'][0] = 0.5
df = pd.DataFrame(data)
M = 50


max_visits_per_nurse = {c:random.randint(7,12) for c in caregivers}

cost_travel = 5.0
cost_caregiver = 20.0
time_cure_patients = {p:0.5*random.randint(1,6) for p in patients}
time_window_earliest = {p:random.randint(8,11) for p in patients}
time_window_latest = {p:random.randint(18,22) for p in patients}
has_skill = {(c,p):1 for c in caregivers for p in patients if 0.95+0.3*random.random()>1}

for c in caregivers:
    has_skill[c,0] =1
    has_skill[0,c] =1

#preference_coefficient = {(p,c):random.random() for c in caregivers for p in patients}
working_hours =  {c:100 for c in caregivers}
max_visits_per_nurse= {'C1': 7, 'C2': 12, 'C3': 8, 'C4': 9}
has_skill = {(c, p1, p2): random.choice([0, 1]) for c in caregivers for p1 in patients for p2 in patients if p1 != p2}
#  {('C1', 2): 1,
#   ('C1', 3): 1,
#   ('C1', 4): 1,
#   ('C1', 5): 1,
#   ('C1', 8): 1,
#   ('C1', 9): 1,
#   ('C1', 10): 1,
#   ('C1', 12): 1,
#   ('C1', 13): 1,
#   ('C1', 14): 1,
#   ('C2', 1): 1,
#   ('C2', 3): 1,
#   ('C2', 4): 1,
#   ('C2', 5): 1,
#   ('C2', 6): 1,
#   ('C2', 7): 1,
#   ('C2', 8): 1,
#   ('C2', 9): 1,
#   ('C2', 11): 1,
#   ('C2', 12): 1,
#   ('C2', 13): 1,
#   ('C2', 14): 1,
#   ('C3', 1): 1,
#   ('C3', 2): 1,
#   ('C3', 3): 1,
#   ('C3', 5): 1,
#   ('C3', 6): 1,
#   ('C3', 8): 1,
#   ('C3', 9): 1,
#   ('C3', 11): 1,
#   ('C3', 12): 1,
#   ('C3', 13): 1,
#   ('C3', 14): 1,
#   ('C4', 1): 1,
#   ('C4', 2): 1,
#   ('C4', 3): 1,
#   ('C4', 4): 1,
#   ('C4', 5): 1,
#   ('C4', 6): 1,
#   ('C4', 7): 1,
#   ('C4', 8): 1,
#   ('C4', 10): 1,
#   ('C4', 11): 1,
#   ('C4', 12): 1,
#   ('C4', 14): 1,
#   ('C1', 0): 1,
#   (0, 'C1'): 1,
#   ('C2', 0): 1,
#   (0, 'C2'): 1,
#   ('C3', 0): 1,
#   (0, 'C3'): 1,
#   ('C4', 0): 1,
#   (0, 'C4'): 1}

time_cure_patients = {1: 3.0,
 2: 0.5,
 3: 3.0,
 4: 0.5,
 5: 1.0,
 6: 2.0,
 7: 0.5,
 8: 1.0,
 9: 2.5,
 10: 0.5,
 11: 2.0,
 12: 1.5,
 13: 2.0,
 14: 3.0}

time_window_earliest = {1: 11,
 2: 9,
 3: 11,
 4: 11,
 5: 9,
 6: 10,
 7: 11,
 8: 10,
 9: 10,
 10: 10,
 11: 9,
 12: 10,
 13: 11,
 14: 8}

time_window_latest= {1: 20,
 2: 21,
 3: 19,
 4: 22,
 5: 18,
 6: 18,
 7: 22,
 8: 19,
 9: 19,
 10: 19,
 11: 18,
 12: 21,
 13: 22,
 14: 18}

skill_level=  {'C1': 3, 'C2': 2, 'C3': 3, 'C4': 3}

data = {'X': [0.500000, 0.302203, 0.753976, 0.458284, 0.633664, 0.586674, 0.551858, 0.633232, 0.580201, 0.557541, 0.724151, 0.438629, 0.265222, 0.835645, 0.042702],
        'y': [0.500000, 0.497371, 0.734364, 0.469632, 0.560095, 0.339246, 0.658934, 0.707274, 0.008343, 0.437174, 0.342705, 0.104080, 0.405842, 0.313082, 0.047175]}


def distance(df,p1,p2):
    return sqrt( (df.loc[p1,'X']-df.loc[p2,'X'])**2 + (df.loc[p1,'y']-df.loc[p2,'y'])**2 )

df = pd.DataFrame(data)
travel_time ={ (p1,p2):round(2*distance(df,p1,p2),2) for p1 in points for p2 in points if p1!=p2}

plt.figure(figsize=(8,8))
plt.scatter(df['X'],df['y'],s=100)


a,b = 1, 0.5
preference_coefficient = {(p,c):a*skill_level[c]+b for c in caregivers for p in patients}
max_customer ={c:sum(1 for p in patients if (c,p) in has_skill) for c in caregivers}

def init_visits(model, c, p):
    if (c,p) in has_skill and has_skill[c,p] == 1:
        model.visits[c,p] = 1
    else:
        model.visits[c,p] = 0
    return model.visits[c,p]

model.visits = pe.Var(caregivers, patients, within=pe.Binary, initialize=init_visits)

#model.visits = pe.Var(caregivers, patients, within=pe.Binary)
def rule_window(model,p):
    return (time_window_earliest[p], time_window_latest[p])


model.start_time = pe.Var(patients, bounds=rule_window, within=pe.PositiveReals)


def init_X(model, c, p1, p2):
    if p1 != p2 and (c,p1) in has_skill and (c,p2) in has_skill and has_skill[c,p1] == 1 and has_skill[c,p2] == 1:
        model.X[c,p1,p2] = 1
    else:
        model.X[c,p1,p2] = 0
    return model.X[c,p1,p2]

model.X = pe.Var(caregivers, points, points, within=pe.Binary, initialize=init_X)

#model.X = pe.Var(caregivers, points, points, within=pe.Binary)
def u_bound(model,c,p):
    return (0,max_customer[c])

model.U = pe.Var(caregivers, patients,bounds=u_bound,within=pe.NonNegativeIntegers)

def dynamic_objective(df):
    df['X'] = [random.random() for i in range(Np+1)]
    df['y'] = [random.random() for i in range(Np+1)]
    df.loc[0,'X']=0.5
    df.loc[0,'y']=0.5
    travel_time ={ (p1,p2):round(2*distance(df,p1,p2),2) for p1 in points for p2 in points if p1!=p2}
    obj = cost_travel * sum(travel_time[p1,p2]*model.X[c,p1,p2] for c in caregivers for p1 in points for p2 in points if p1!=p2) + \
          cost_caregiver* sum(skill_level[c]*model.visits[c,p] for c in caregivers for p in patients) - \
          sum(preference_coefficient[p,c]*model.visits[c,p] for c in caregivers for p in patients)
    return obj
model.dynamic_objective = pe.Param(initialize=dynamic_objective(df), mutable=True)

def obj_rule(model):
    return model.dynamic_objective
model.combined_objective = pe.Objective(rule=obj_rule, sense=pe.minimize)
  # return cost_travel * sum(travel_time[p1,p2]*model.X[c,p1,p2] for c in caregivers for p1 in points for p2 in points if p1!=p2) + \
  #           cost_caregiver * sum(skill_level[c]*model.visits[c,p] for c in caregivers for p in patients) - \
  #           sum(preference_coefficient[p,c]*model.visits[c,p] for c in caregivers for p in patients)
model.combined_objective = pe.Objective(rule=obj_rule, sense=pe.minimize)

if hasattr(model, 'visits_cons_index'):
    del model.visits_cons_index
def rule_visit(model,p):
    return sum(model.visits[c,p] for c in caregivers) == 1
model.visits_cons = pe.Constraint(patients, rule=rule_visit)

if hasattr(model, 'max_visit_cons_index'):
    del model.max_visit_cons_index

def rule_max_visit(model,c):
    return sum(model.visits[c,p] for p in patients) <= max_visits_per_nurse[c]
model.max_visit_cons = pe.Constraint(caregivers, rule=rule_max_visit)

# if hasattr(model, 'skill'):
#     del model.skill
if hasattr(model, 'skill_index'):
    model.del_component(model.skill_index)
if hasattr(model, 'skill_index_0'):
    model.del_component(model.skill_index_0)
if hasattr(model, 'skill_index_1'):
    model.del_component(model.skill_index_1)

def rule_skill(model,caregivers, patients):
    #return model.visits[c,p] <= has_skill[c,p]
    if (caregivers, patients) in has_skill:
        return model.visits[caregivers, patients] <= has_skill[caregivers, patients]
    else:
        return pe.Constraint.Skip

model.skill = pe.Constraint(caregivers, patients, rule=rule_skill)

if hasattr(model, 'start_time_rule_index'):
    model.del_component(model.start_time_rule_index)
if hasattr(model, 'start_time_rule_index_0'):
    model.del_component(model.start_time_rule_index_0)
if hasattr(model, 'start_time_rule_index_1'):
    model.del_component(model.start_time_rule_index_1)
if hasattr(model, 'start_time_rule_index_2'):
    model.del_component(model.start_time_rule_index_2)

def rule_start_time(model, c, p):
    return model.start_time[p] >= time_window_earliest[p] + sum(
        model.X[c, p1, p] * (time_cure_patients.get(p1, 0) + travel_time.get((p1, p), 0))
        for p1 in points if p1 != p)
model.start_time_rule = pe.Constraint(caregivers, patients, rule=rule_start_time)

if hasattr(model, 'end_time_rule_index'):
    model.del_component(model.end_time_rule_index)
if hasattr(model, 'end_time_rule_index_0'):
    model.del_component(model.end_time_rule_index_0)
if hasattr(model, 'end_time_rule_index_1'):
    model.del_component(model.end_time_rule_index_1)
if hasattr(model, 'end_time_rule_index_2'):
    model.del_component(model.end_time_rule_index_2)

def rule_end_time(model, c, p):
    return model.start_time[p] <= time_window_latest[p] - sum(model.X[c, p, p2] * (time_cure_patients[p] + travel_time[p, p2]) for p2 in points if p2 != p)
model.end_time_rule = pe.Constraint(caregivers, patients, rule=rule_end_time)

if hasattr(model, 'working_hours_rule_index'):
    model.del_component(model.working_hours_rule_index)
if hasattr(model, 'working_hours_rule_index_0'):
    model.del_component(model.working_hours_rule_index_0)
if hasattr(model, 'working_hours_rule_index_1'):
    model.del_component(model.working_hours_rule_index_1)
if hasattr(model, 'working_hours_rule_index_2'):
    model.del_component(model.working_hours_rule_index_2)
def rule_working_hours(model,caregivers):
    return sum(model.X[caregivers, p1, p2] * (time_cure_patients.get(p1, 0) + travel_time.get((p1, p2), 0))
               for p1 in points for p2 in points if p1 != p2) <= working_hours[caregivers]
model.working_hours_rule = pe.Constraint(caregivers, rule=rule_working_hours)


if hasattr(model, 'X1_index'):
    model.del_component(model.X1_index)
if hasattr(model, 'X1_index_0'):
    model.del_component(model.X1_index_0)
if hasattr(model, 'X1_index_1'):
    model.del_component(model.X1_index_1)
if hasattr(model, 'X1_index_2'):
    model.del_component(model.X1_index_2)
def rule_X1(model,c,p):
    return model.X[c,0,p] == model.visits[c,p]

model.X1 = pe.Constraint(caregivers, patients, rule=rule_X1)

if hasattr(model, 'X2_index'):
    model.del_component(model.X2_index)
if hasattr(model, 'X2_index_0'):
    model.del_component(model.X2_index_0)
if hasattr(model, 'X2_index_1'):
    model.del_component(model.X2_index_1)
if hasattr(model, 'X2_index_2'):
    model.del_component(model.X2_index_2)
def rule_X2(model,c,p):
    return model.X[c,p,0] == model.visits[c,p]

model.X2 = pe.Constraint(caregivers, patients, rule=rule_X2)

if hasattr(model, 'X3_index'):
    model.del_component(model.X3_index)
if hasattr(model, 'X3_index_0'):
    model.del_component(model.X3_index_0)
if hasattr(model, 'X3_index_1'):
    model.del_component(model.X3_index_1)
if hasattr(model, 'X3_index_2'):
    model.del_component(model.X3_index_2)
def rule_X3(model,c,p):
    return sum(model.X[c,p1,p] for p1 in points if p1!=p) == model.visits[c,p]
model.X3 = pe.Constraint(caregivers, patients, rule=rule_X3)

if hasattr(model, 'X4_index'):
    model.del_component(model.X4_index)
if hasattr(model, 'X4_index_0'):
    model.del_component(model.X4_index_0)
if hasattr(model, 'X4_index_1'):
    model.del_component(model.X4_index_1)
if hasattr(model, 'X4_index_2'):
    model.del_component(model.X4_index_2)
def rule_X4(model,c,p):
    return sum(model.X[c,p,p2] for p2 in points if p2!=p) == model.visits[c,p]
model.X4 = pe.Constraint(caregivers, patients, rule=rule_X4)


if hasattr(model, 'U1_index'):
    model.del_component(model.U1_index)
if hasattr(model, 'U1_index_0'):
    model.del_component(model.U1_index_0)
if hasattr(model, 'U1_index_1'):
    model.del_component(model.U1_index_1)
if hasattr(model, 'U1_index_2'):
    model.del_component(model.U1_index_2)
def rule_U1(model,c,p):
    return model.U[c,p] >= 1
model.U1 = pe.Constraint(caregivers, patients, rule=rule_U1)

if hasattr(model, 'U2_index'):
    model.del_component(model.U2_index)
if hasattr(model, 'U2_index_0'):
    model.del_component(model.U2_index_0)
if hasattr(model, 'U2_index_1'):
    model.del_component(model.U2_index_1)
if hasattr(model, 'U2_index_2'):
    model.del_component(model.U2_index_2)
def rule_U2(model,c,p1,p2):
    return model.U[c,p2] >= model.U[c,p1] + 1 - M*(1-model.X[c,p1,p2])

model.U2 = pe.Constraint(caregivers, patients, patients, rule=rule_U2)

# Define a function to generate a random initial solution
def random_initialization(model):
    # Create a dictionary to store the initial values of the variables
    init_values = {}
    # Loop over the caregivers
    for c in caregivers:
        # Initialize the number of visits for the current caregiver
        num_visits = 0
        # Loop over the patients
        for p in patients:
            # Randomly assign the patient to the caregiver with some probability
            if (c, p) in has_skill and random.random() < 0.5 and has_skill[c,p] == 1 and num_visits < max_visits_per_nurse[c]:
                # Set the visit variable to 1 using tuple (c, p) as the key
                init_values[(c, p)] = 1
                # Increment the number of visits for the current caregiver
                num_visits += 1
            else:
                # Set the visit variable to 0 using tuple (c, p) as the key
                init_values[(c, p)] = 0
        # Loop over the points
        for p1 in points:
            for p2 in points:
                # Check if the caregiver travels from p1 to p2 and visits both points
                if p1 != p2 and init_values.get((c, p1), 0) == 1 and init_values.get((c, p2), 0) == 1:
                    # Set the X variable to 1 using tuple (c, p1, p2) as the key
                    init_values[(c, p1, p2)] = 1
                else:
                    # Set the X variable to 0 using tuple (c, p1, p2) as the key
                    init_values[(c, p1, p2)] = 0
        # Initialize a list to store the visited patients by the current caregiver
        visited_patients = []

        default_start_time = 0
        for p in visited_patients:
          if model.start_time[p].value is None:
              model.start_time[p].value = default_start_time
          # Loop over the patients
        for p in patients:
            # If the caregiver visits the patient, append the patient to the list
            if init_values.get((c, p), 0) == 1:
                visited_patients.append(p)
        # Sort the visited patients by their start time
        visited_patients.sort(key=lambda p: model.start_time[p].value if model.start_time[p].value is not None else default_start_time)  # Assuming start_time has been initialized
        # Loop over the visited patients
        for i, p in enumerate(visited_patients):
            # Set the U variable to the position of the patient in the route using tuple (c, p) as the key
            init_values[(c, p)] = i + 1
    # Return the dictionary of initial values
    return init_values

# Define a function to create a neighborhood of a given solution
def create_neighborhood(model, solution, k):
    neighbors = []
    for i in range(k):
        neighbor = solution.copy()
        c1, c2 = random.sample(caregivers, 2)
        p1, p2 = random.sample(patients, 2)

        # Swap visits between two caregivers for two patients
        temp = neighbor.get((c1, p1), 0)
        neighbor[(c1, p1)] = neighbor.get((c2, p2), 0)
        neighbor[(c2, p2)] = temp

        # Ensure the X variables are correctly updated based on the new visits
        for c in caregivers:
            for p1 in points:
                for p2 in points:
                    if p1 != p2 and neighbor.get((c, p1), 0) == 1 and neighbor.get((c, p2), 0) == 1:
                        neighbor[(c, p1, p2)] = 1
                    else:
                        neighbor[(c, p1, p2)] = 0

        # Correctly update the U variables based on the sorted order of visited patients
        for c in caregivers:
            visited_patients = [p for p in patients if neighbor.get((c, p), 0) == 1]
            # Sort the visited patients based on some criteria, e.g., patient ID for simplicity
            visited_patients.sort()
            for i, p in enumerate(visited_patients):
                neighbor[(c, p)] = i + 1  # Update U variable based on the order

        neighbors.append(neighbor)
    return neighbors


# def evaluate_solution(model, solution):

#     for key, val in solution.items():
#         if isinstance(key, tuple):
#             if len(key) == 2:  # For model.visits
#                 model.visits[key].set_value(val)
#             elif len(key) == 3:  # For model.X
#                 model.X[key].set_value(val)
#             # Add handling for model.U if necessary
#         else:
#             print(f"Unexpected key structure: {key}")


#     model.combined_objective.expr = obj_rule(model)
#     obj_value = pe.value(model.combined_objective)
    

#     travel_cost = sum(travel_time[p1,p2]*model.X[c,p1,p2].value for c in caregivers for p1 in points for p2 in points if p1!=p2)
#     caregiver_cost = sum(skill_level[c]*model.visits[c,p].value for c in caregivers for p in patients)
#     preference_cost = sum(preference_coefficient[p,c]*model.visits[c,p].value for c in caregivers for p in patients)
#     print(f"Travel Cost: {travel_cost}, Caregiver Cost: {caregiver_cost}, Preference Cost: {preference_cost}")
    
#     return obj_value

def feasible_initialization(model):
    for c in caregivers:
        for p in patients:  
            if (c, p) in has_skill and has_skill[c, p] == 1:
                model.visits[c, p].set_value(1)
            else:
                model.visits[c, p].set_value(0)

    for c in caregivers:
        for p1 in points:
            for p2 in points:
                if p1 != p2:
                    if p1 == 0 or p2 == 0 or (model.visits[c, p1].value == 1 and model.visits[c, p2].value == 1):
                        model.X[c, p1, p2].set_value(1)
                    else:
                        model.X[c, p1, p2].set_value(0)
                        
    for c in caregivers:
        visit_order = 1
        for p in sorted(patients):
            if model.visits[c, p].value == 1:
                model.U[c, p].set_value(visit_order)
                visit_order += 1


def evaluate_solution(model,solution):
    model.combined_objective.expr = obj_rule(model)
    return pe.value(model.combined_objective)





def improve_solution(model, solution):
    # Create a copy of the current solution
    improved_solution = solution.copy()
    # Evaluate the current solution
    current_obj = evaluate_solution(model, solution)
    # Set a flag to indicate if improvement is possible
    improvement = True
    # Loop until no improvement is possible
    while improvement:
        # Set the flag to False
        improvement = False
        # Create a neighborhood of the current solution
        neighbors = create_neighborhood(model, improved_solution, k=10)
        # Loop over the neighbors
        for neighbor in neighbors:
            # Evaluate the neighbor solution
            neighbor_obj = evaluate_solution(model, neighbor)
            # If the neighbor solution is better than the current solution
            if neighbor_obj < current_obj:
                # Set the current solution to the neighbor solution
                improved_solution = neighbor
                # Set the current objective value to the neighbor objective value
                current_obj = neighbor_obj
                # Set the flag to True
                improvement = True
                # Break the loop
                break
    # Return the improved solution
    return improved_solution

# Define a function to select the best solution among a list of solutions
def select_best_solution(model, solutions):
    # Initialize the best solution to None
    best_solution = None
    # Initialize the best objective value to infinity
    best_obj = float('inf')
    # Loop over the solutions
    for solution in solutions:
        # Evaluate the solution
        obj_value = evaluate_solution(model, solution)
        # If the solution is better than the best solution
        if obj_value < best_obj:
            # Set the best solution to the solution
            best_solution = solution
            # Set the best objective value to the objective value
            best_obj = obj_value
    # Return the best solution
    return best_solution

# Define a function to implement the variable neighborhood search algorithm
def variable_neighborhood_search(model, max_iter=100, k_max=10):
    # Generate a random initial solution
    current_solution = random_initialization(model)
    # Set the current iteration to 0
    current_iter = 0
    # Set the current neighborhood size to 1
    current_k = 1
    # Loop until the maximum iteration is reached or the neighborhood size exceeds the maximum size
    while current_iter < max_iter and current_k <= k_max:
        # Improve the current solution using a local search method
        improved_solution = improve_solution(model, current_solution)
        # Evaluate the current solution and the improved solution
        current_obj = evaluate_solution(model, current_solution)
        improved_obj = evaluate_solution(model, improved_solution)
        # If the improved solution is better than the current solution
        if improved_obj < current_obj:
            # Set the current solution to the improved solution
            current_solution = improved_solution
            # Reset the neighborhood size to 1
            current_k = 1
        else:
            # Increase the neighborhood size by 1
            current_k += 1
        # Increase the iteration by 1
        current_iter += 1
    # Return the current solution
    return current_solution

solution = variable_neighborhood_search(model)
#solution = variable_neighborhood_search(instance)
feasible_initialization(model)

# Example of how to use the improved evaluation function
current_obj_value = evaluate_solution(model,solution)
print(f"Current Objective Value: {current_obj_value}")
if solution is not None:
    print("Solution Found!")
    print('OF = ', evaluate_solution(model, solution))
    KOLOR = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
            "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
            "#8c6d31", "#9c9ede", "#637939", "#e7cb94", "#b5cf6b",
            "#cedb9c", "#c7c7c7", "#bd9e39", "#e7969c", "#7b4173"]

    print("\nVariable Values:")

    plt.figure(figsize=(9,9))
    plt.scatter(df['X'],df['y'],s=700, c='k')
    for n in points:
        plt.text(df.loc[n,'X']-0.01,df.loc[n,'y'],s=str(n), fontsize=14, c='w', zorder=2, fontweight='bold')
    for c in caregivers:
        visited_patients = [p for p in patients if solution.get((c, p), 0) == 1]
        # Sort the visited patients by their start time, using a large default value for patients without a start time
        visited_patients.sort(key=lambda p: model.start_time[p].value if model.start_time[p].value is not None else float('inf'))
        for i, p in enumerate(visited_patients):
            if i == 0:
                x0,y0 = df.loc[0,'X'] ,df.loc[0,'y']
                x1,y1 = df.loc[p,'X'] ,df.loc[p,'y']
                plt.plot([x0,x1],[y0,y1], c = KOLOR[caregivers.index(c)], lw=5, zorder=-1)
                print(f"nurse {c}, travels from 0 to {p}")
            elif i == len(visited_patients) - 1:
                x0,y0 = df.loc[p,'X'] ,df.loc[p,'y']
                x1,y1 = df.loc[0,'X'] ,df.loc[0,'y']
                plt.plot([x0,x1],[y0,y1], c = KOLOR[caregivers.index(c)], lw=5, zorder=-1)
                print(f"nurse {c}, travels from {p} to 0")
            else:
                p_prev = visited_patients[i-1]
                x0,y0 = df.loc[p_prev,'X'] ,df.loc[p_prev,'y']
                x1,y1 = df.loc[p,'X'] ,df.loc[p,'y']
                plt.plot([x0,x1],[y0,y1], c = KOLOR[caregivers.index(c)], lw=5, zorder=-1)
                print(f"nurse {c}, travels from {p_prev} to {p}")
    plt.grid()
    plt.show()
else:
    print("No Solution Found.")

import pandas as pd


print("\nTreatment Start Times:")
start_times = {p: model.start_time[p].value for p in patients}
start_times_df = pd.DataFrame(list(start_times.items()), columns=['Patient', 'Start Time'])
print(start_times_df)

# Nurse Routing
print("\nNurse Routing:")
routes = []
for c in caregivers:
    route = [0]  # Starting from the depot
    visited_patients = [p for p in patients if solution.get((c, p), 0) == 1]
    visited_patients.sort(key=lambda p: model.start_time[p].value if model.start_time[p].value is not None else float('inf'))
    route.extend(visited_patients)
    route.append(0)  # Returning to the depot
    routes.append((c, ' -> '.join(map(str, route))))
routes_df = pd.DataFrame(routes, columns=['Nurse', 'Route'])
print(routes_df)

# Nurse Allocation to Patients
print("\nNurse Allocation to Patients:")
allocations = [(c, p) for c in caregivers for p in patients if solution.get((c, p), 0) == 1]
allocations_df = pd.DataFrame(allocations, columns=['Nurse', 'Patient'])
print(allocations_df)
print()
