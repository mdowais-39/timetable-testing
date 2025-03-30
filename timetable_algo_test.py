import random
from ortools.sat.python import cp_model
from deap import base, creator, tools, algorithms
from collections import defaultdict

# ==== Mock Data (Replace with Database) ====
courses = [
    {"id": "CS101", "lectures": 3, "tutorials": 1, "practicals": 2, "professor": "P1", "assistants": ["A1"], "students": 50}
]
classrooms = [
    {"id": "R1", "type": "lecture", "capacity": 60},
    {"id": "R2", "type": "lab", "capacity": 30}
]
timeslots = ["Mon_9AM", "Mon_10AM", "Tue_2PM", "Wed_3PM"]
faculty = ["P1", "A1"]

# ==== Genetic Algorithm Setup ====
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def create_gene():
    """Create a random class session (gene)."""
    course = random.choice(courses)
    session_type = random.choice(["lecture", "tutorial", "practical"])
    classroom = next(c for c in classrooms if (
        (session_type == "practical" and c["type"] == "lab") or
        (session_type != "practical" and c["type"] == "lecture")
    ))
    return {
        "course": course["id"],
        "session_type": session_type,
        "professor": course["professor"],
        "assistants": course["assistants"],
        "classroom": classroom["id"],
        "timeslot": random.choice(timeslots)
    }

toolbox = base.Toolbox()
toolbox.register("gene", create_gene)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene, n=10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ==== Fitness Function ====
def evaluate(individual):
    score = 100
    faculty_slots = defaultdict(set)
    classroom_slots = defaultdict(set)
    
    for gene in individual:
        # Penalize faculty overlaps
        for instructor in [gene["professor"]] + gene["assistants"]:
            if gene["timeslot"] in faculty_slots[instructor]:
                score -= 10
            faculty_slots[instructor].add(gene["timeslot"])
        # Penalize classroom overlaps
        if gene["timeslot"] in classroom_slots[gene["classroom"]]:
            score -= 15
        classroom_slots[gene["classroom"]].add(gene["timeslot"])
    return (score,)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# ==== Constraint Repair (CSP) ====
def repair_schedule(individual):
    model = cp_model.CpModel()
    vars = {i: model.NewIntVar(0, len(timeslots)-1, f"gene_{i}") for i in range(len(individual))}
    
    # No faculty overlaps
    faculty_slots = defaultdict(list)
    for i, gene in enumerate(individual):
        for instructor in [gene["professor"]] + gene["assistants"]:
            faculty_slots[instructor].append(vars[i])
    for slots in faculty_slots.values():
        model.AddAllDifferent(slots)
    
    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status == cp_model.OPTIMAL:
        for i, gene in enumerate(individual):
            gene["timeslot"] = timeslots[solver.Value(vars[i])]
    return individual

# ==== Main Algorithm ====
def generate_timetable():
    population = toolbox.population(n=50)
    for gen in range(100):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.2)
        # Repair offspring using CSP
        repaired_offspring = [repair_schedule(ind) for ind in offspring]
        population = toolbox.select(repaired_offspring + population, k=50)
    return tools.selBest(population, k=1)[0]

# ==== Run ====
best = generate_timetable()
print("Best Timetable:")
for gene in best:
    print(f"{gene['timeslot']}: {gene['course']} ({gene['session_type']}) in {gene['classroom']}")