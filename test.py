#!pip install pyomo
import pyomo.environ as pe
import os
import random
import pandas as pd
from math import sqrt
import math
import matplotlib.pyplot as plt
os.environ['NEOS_EMAIL'] = 'XXXXXX@gmail.com'
global p,p1,p2,working_hours,M
model = pe.ConcreteModel()

Np = 14
Nc = 4
patients = [p for p in range(1,1+Np)]
points = [0]+patients
print(points)
caregivers = [f"C{c}" for c in range(1,1+Nc)]
skill_level = {c:random.randint(1,5) for c in caregivers }



df = pd.DataFrame()
df['x'] = [random.random() for i in range(Np+1)]
df['y'] = [random.random() for i in range(Np+1)]
df.loc[0,'x']=0.5
df.loc[0,'y']=0.5

M = 50
# Parameters


max_visits_per_nurse = {c:random.randint(7,12) for c in caregivers}

cost_travel = 5.0
cost_caregiver = 20.0 # if skill level is 1 , will be multiplied by the skill level
#  look at here --->    visit_cost = cost_caregiver* sum(skill_level[c]*model.visits[c,p] for c in caregivers for p in patients)


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
has_skill = {('C1', 2): 1,
  ('C1', 3): 1,
  ('C1', 4): 1,
  ('C1', 5): 1,
  ('C1', 8): 1,
  ('C1', 9): 1,
  ('C1', 10): 1,
  ('C1', 12): 1,
  ('C1', 13): 1,
  ('C1', 14): 1,
  ('C2', 1): 1,
  ('C2', 3): 1,
  ('C2', 4): 1,
  ('C2', 5): 1,
  ('C2', 6): 1,
  ('C2', 7): 1,
  ('C2', 8): 1,
  ('C2', 9): 1,
  ('C2', 11): 1,
  ('C2', 12): 1,
  ('C2', 13): 1,
  ('C2', 14): 1,
  ('C3', 1): 1,
  ('C3', 2): 1,
  ('C3', 3): 1,
  ('C3', 5): 1,
  ('C3', 6): 1,
  ('C3', 8): 1,
  ('C3', 9): 1,
  ('C3', 11): 1,
  ('C3', 12): 1,
  ('C3', 13): 1,
  ('C3', 14): 1,
  ('C4', 1): 1,
  ('C4', 2): 1,
  ('C4', 3): 1,
  ('C4', 4): 1,
  ('C4', 5): 1,
  ('C4', 6): 1,
  ('C4', 7): 1,
  ('C4', 8): 1,
  ('C4', 10): 1,
  ('C4', 11): 1,
  ('C4', 12): 1,
  ('C4', 14): 1,
  ('C1', 0): 1,
  (0, 'C1'): 1,
  ('C2', 0): 1,
  (0, 'C2'): 1,
  ('C3', 0): 1,
  (0, 'C3'): 1,
  ('C4', 0): 1,
  (0, 'C4'): 1}

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

data = {'x': [0.500000, 0.302203, 0.753976, 0.458284, 0.633664, 0.586674, 0.551858, 0.633232, 0.580201, 0.557541, 0.724151, 0.438629, 0.265222, 0.835645, 0.042702],
        'y': [0.500000, 0.497371, 0.734364, 0.469632, 0.560095, 0.339246, 0.658934, 0.707274, 0.008343, 0.437174, 0.342705, 0.104080, 0.405842, 0.313082, 0.047175]}

# datas = {None: {
#     'Np': {None: 14},
#     'Nc': {None: 4},
#     'max_visits_per_nurse': {c:random.randint(7,12) for c in caregivers},
#     'cost_travel': {None: 5.0},
#     'cost_caregiver': {None: 20.0},
#     'time_cure_patients': {p:0.5*random.randint(1,6) for p in patients},
#     'time_window_earliest': {p:random.randint(8,11) for p in patients},
#     'time_window_latest': {p:random.randint(18,22) for p in patients},
#     'skill_level': {c:random.randint(1,5) for c in caregivers},
#     'has_skill': {(c,p):1 for c in caregivers for p in patients if 0.95+0.3*random.random()>1},
#     'data': {None: data}
# }}

# instance = model.create_instance(datas)

df = pd.DataFrame(data)
travel_time ={ (p1,p2):round(2*dist(df,p1,p2),2) for p1 in points for p2 in points if p1!=p2}


def dist(df,p1,p2):
    return sqrt( (df.loc[p1,'x']-df.loc[p2,'x'])**2 + (df.loc[p1,'y']-df.loc[p2,'y'])**2 )

a,b = 1, 0.5
preference_coefficient = {(p,c):a*skill_level[c]+b for c in caregivers for p in patients}
max_customer ={c:sum(1 for p in patients if (c,p) in has_skill) for c in caregivers}

model.visits = pe.Var(caregivers, patients, within=pe.Binary) 
def rule_window(model,p):
    return (time_window_earliest[p], time_window_latest[p])


model.start_time = pe.Var(patients, bounds=rule_window, within=pe.PositiveReals) 

model.X = pe.Var(caregivers, points, points, within=pe.Binary)
def u_bound(model,c,p):
    return (0,max_customer[c])

model.U = pe.Var(caregivers, patients,bounds=u_bound,within=pe.NonNegativeIntegers)


def dynamic_objective(df):
    # Update the dataframe with some changes
    # For example, add or remove some nodes, change some parameters, etc.
    # Here we just randomly change the x and y coordinates of the nodes
    df['x'] = [random.random() for i in range(Np+1)]
    df['y'] = [random.random() for i in range(Np+1)]
    df.loc[0,'x']=0.5
    df.loc[0,'y']=0.5
    # Calculate the new travel time based on the updated dataframe
    travel_time ={ (p1,p2):round(2*dist(df,p1,p2),2) for p1 in points for p2 in points if p1!=p2}
    # Define the objective function based on the updated travel time and other parameters
    obj = cost_travel * sum(travel_time[p1,p2]*model.X[c,p1,p2] for c in caregivers for p1 in points for p2 in points if p1!=p2) + \
          cost_caregiver* sum(skill_level[c]*model.visits[c,p] for c in caregivers for p in patients) - \
          sum(preference_coefficient[p,c]*model.visits[c,p] for c in caregivers for p in patients)
    return obj

# Define the model objective using the dynamic objective parameter
def obj_rule(model):
    return model.dynamic_objective

# Define the constraints
def rule_visit(model,p):
    return sum(model.visits[c,p] for c in caregivers) == 1

def rule_max_visit(model,c):
    return sum(model.visits[c,p] for p in patients) <= max_visits_per_nurse[c]

def rule_skill(model,c,p):
    return model.visits[c,p] <= has_skill[c,p]


def rule_start_time(model,c,p):
    return model.start_time[p] >= time_window_earliest[p] + sum(model.X[c,p1,p] for p1 in points if p1!=p)*(time_cure_patients[p1]+travel_time[p1,p])


def rule_end_time(model,c,p):
    return model.start_time[p] <= time_window_latest[p] - sum(model.X[c,p,p2] for p2 in points if p2!=p)*(time_cure_patients[p]+travel_time[p,p2])


def rule_working_hours(model,c):
    return sum(model.X[c,p1,p2]*(time_cure_patients[p1]+travel_time[p1,p2]) for p1 in points for p2 in points if p1!=p2) <= working_hours[c]


def rule_X1(model,c,p):
    return model.X[c,0,p] == model.visits[c,p]


# Define the X2 constraint
def rule_X2(model,c,p):
    return model.X[c,p,0] == model.visits[c,p]


# Define the X3 constraint
def rule_X3(model,c,p):
    return sum(model.X[c,p1,p] for p1 in points if p1!=p) == model.visits[c,p]


# Define the X4 constraint
def rule_X4(model,c,p):
    return sum(model.X[c,p,p2] for p2 in points if p2!=p) == model.visits[c,p]


# Define the U1 constraint
def rule_U1(model,c,p):
    return model.U[c,p] >= 1


# Define the U2 constraint
def rule_U2(model,c,p1,p2):
    return model.U[c,p2] >= model.U[c,p1] + 1 - M*(1-model.X[c,p1,p2])


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
            if random.random() < 0.5 and has_skill[c,p] == 1 and num_visits < max_visits_per_nurse[c]:
                # Set the visit variable to 1
                init_values[model.visits[c,p]] = 1
                # Increment the number of visits for the current caregiver
                num_visits += 1
            else:
                # Set the visit variable to 0
                init_values[model.visits[c,p]] = 0
        # Loop over the points
        for p1 in points:
            for p2 in points:
                # Set the X variable to 1 if the caregiver travels from p1 to p2 and visits both points
                if p1 != p2 and init_values[model.visits[c,p1]] == 1 and init_values[model.visits[c,p2]] == 1:
                    init_values[model.X[c,p1,p2]] = 1
                else:
                    # Set the X variable to 0 otherwise
                    init_values[model.X[c,p1,p2]] = 0
        # Initialize a list to store the visited patients by the current caregiver
        visited_patients = []
        # Loop over the patients
        for p in patients:
            # If the caregiver visits the patient, append the patient to the list
            if init_values[model.visits[c,p]] == 1:
                visited_patients.append(p)
        # Sort the visited patients by their start time
        visited_patients.sort(key=lambda p: init_values[model.start_time[p]])
        # Loop over the visited patients
        for i, p in enumerate(visited_patients):
            # Set the U variable to the position of the patient in the route
            init_values[model.U[c,p]] = i + 1
    # Return the dictionary of initial values
    return init_values

# Define a function to create a neighborhood of a given solution
# Define a function to create a neighborhood of a given solution
def create_neighborhood(model, solution, k):
    # Create a list to store the neighboring solutions
    neighbors = []
    # Loop k times
    for i in range(k):
        # Create a copy of the current solution
        neighbor = solution.copy()
        # Randomly select two caregivers
        c1, c2 = random.sample(caregivers, 2)
        # Randomly select two patients
        p1, p2 = random.sample(patients, 2)
        # Swap the visit variables of the selected caregivers and patients
        neighbor[model.visits[c1,p1]], neighbor[model.visits[c2,p2]] = neighbor[model.visits[c2,p2]], neighbor[model.visits[c1,p1]]
        # Update the X and U variables accordingly
        # Loop over the caregivers
        for c in [c1, c2]:
            # Loop over the points
            for p1 in points:
                for p2 in points:
                    # Set the X variable to 1 if the caregiver travels from p1 to p2 and visits both points
                    if p1 != p2 and neighbor[model.visits[c,p1]] == 1 and neighbor[model.visits[c,p2]] == 1:
                        neighbor[model.X[c,p1,p2]] = 1
                    else:
                        # Set the X variable to 0 otherwise
                        neighbor[model.X[c,p1,p2]] = 0
            # Initialize a list to store the visited patients by the current caregiver
            visited_patients = []
            # Loop over the patients
            for p in patients:
                # If the caregiver visits the patient, append the patient to the list
                if neighbor[model.visits[c,p]] == 1:
                    visited_patients.append(p)
            # Sort the visited patients by their position in the route
            visited_patients.sort(key=lambda p: neighbor[model.U[c,p]])
            # Loop over the visited patients
            for i, p in enumerate(visited_patients):
                # Set the U variable to the position of the patient in the route
                neighbor[model.U[c,p]] = i + 1
        # Append the neighbor to the list of neighbors
        neighbors.append(neighbor)
    # Return the list of neighbors
    return neighbors

# Define a function to evaluate a given solution
def evaluate_solution(model, solution):
    # Set the value of the variables according to the solution
    for var, val in solution.items():
        var.set_value(val)
    # Calculate the value of the objective function
    obj_value = pe.value(model.combined_objective)
    # Return the objective value
    return obj_value

# Define a function to improve a given solution using a local search method
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
#model.del_component(model.X)

#model.start_time = pe.Var(patients, bounds=rule_window, within=pe.PositiveReals) # Start time of the visit for each patient

model.dynamic_objective = pe.Param(initialize=dynamic_objective(df), mutable=True)
model.combined_objective = pe.Objective(rule=obj_rule, sense=pe.minimize)
model.visit = pe.Constraint(patients, rule=rule_visit)
model.max_visit = pe.Constraint(caregivers, rule=rule_max_visit)
model.skill = pe.Constraint(caregivers, patients, rule=rule_skill)
model.start_time_rule = pe.Constraint(caregivers, patients, rule=rule_start_time)
model.end_time_rule = pe.Constraint(caregivers, patients, rule=rule_end_time)
model.working_hours_rule = pe.Constraint(caregivers, rule=rule_working_hours)
model.X1 = pe.Constraint(caregivers, patients, rule=rule_X1)
model.X2 = pe.Constraint(caregivers, patients, rule=rule_X2)
model.X3 = pe.Constraint(caregivers, patients, rule=rule_X3)
model.X4 = pe.Constraint(caregivers, patients, rule=rule_X4)
model.U1 = pe.Constraint(caregivers, patients, rule=rule_U1)
model.U2 = pe.Constraint(caregivers, patients, patients, rule=rule_U2)




# Solve the model using the variable neighborhood search algorithm

solution = variable_neighborhood_search(model)
#solution = variable_neighborhood_search(instance)
if solution is not None:
    print("Solution Found!")
    print('OF = ', evaluate_solution(model, solution))
    # Define a list of colors for plotting
    KOLOR = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
            "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
            "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
            "#8c6d31", "#9c9ede", "#637939", "#e7cb94", "#b5cf6b",
            "#cedb9c", "#c7c7c7", "#bd9e39", "#e7969c", "#7b4173"]

    print("\nVariable Values:")
    # Plot the points and their labels
    plt.figure(figsize=(9,9))
    plt.scatter(df['x'],df['y'],s=700, c='k')
    for n in points:
        plt.text(df.loc[n,'x']-0.01,df.loc[n,'y'],s=str(n), fontsize=14, c='w', zorder=2, fontweight='bold')
    # Loop over the caregivers
    for c in caregivers:
        # Get the list of visited patients by the current caregiver
        visited_patients = [p for p in patients if solution[model.visits[c,p]] == 1]
        # Sort the visited patients by their position in the route
        visited_patients.sort(key=lambda p: solution[model.U[c,p]])
        # Loop over the visited patients
        for i, p in enumerate(visited_patients):
            # If it is the first patient, draw a line from the depot to the patient
            if i == 0:
                x0,y0 = df.loc[0,'x'] ,df.loc[0,'y']
                x1,y1 = df.loc[p,'x'] ,df.loc[p,'y']
                plt.plot([x0,x1],[y0,y1], c = KOLOR[caregivers.index(c)], lw=5, zorder=-1)
                print(f"nurse {c}, travels from 0 to {p}")
            # If it is the last patient, draw a line from the patient to the depot
            elif i == len(visited_patients) - 1:
                x0,y0 = df.loc[p,'x'] ,df.loc[p,'y']
                x1,y1 = df.loc[0,'x'] ,df.loc[0,'y']
                plt.plot([x0,x1],[y0,y1], c = KOLOR[caregivers.index(c)], lw=5, zorder=-1)
                print(f"nurse {c}, travels from {p} to 0")
            # Otherwise, draw a line from the previous patient to the current patient
            else:
                p_prev = visited_patients[i-1]
                x0,y0 = df.loc[p_prev,'x'] ,df.loc[p_prev,'y']
                x1,y1 = df.loc[p,'x'] ,df.loc[p,'y']
                plt.plot([x0,x1],[y0,y1], c = KOLOR[caregivers.index(c)], lw=5, zorder=-1)
                print(f"nurse {c}, travels from {p_prev} to {p}")
    # Show the grid and the plot
    plt.grid()
    plt.show()
else:
    print("No Solution Found.")
