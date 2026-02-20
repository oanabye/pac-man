# pac-man

A Python-based implementation of classic Artificial Intelligence algorithms applied to the Pac-Man game environment. Developed as part of the **Artificial Intelligence** course in my 3rd year of Bachelor's degree at a Technical University.

The game framework and environment were provided as part of the coursework. All search algorithms, heuristics, and agent logic were implemented by me.

---

## 🧠 Overview

The goal of this project was to build intelligent agents capable of navigating a maze, collecting food, and handling adversarial ghost agents — using well-established AI techniques.

Pac-Man operates in a discrete 2D grid where every move is a state transition. The agent must make decisions based on the current game state, sometimes under adversarial conditions.

---

## ⚙️ Implemented Algorithms

### Single-Agent Search
Implemented in `search.py` and `searchAgents.py`:

- **Depth-First Search (DFS)**
- **Breadth-First Search (BFS)**
- **Uniform-Cost Search (UCS)**
- **A\* Search** with admissible and consistent heuristics

These allow Pac-Man to find valid paths through the maze, compute optimal routes, and collect all food efficiently.

### Advanced Search Problems

- **Corners Problem** — tracking and visiting all four corners of the maze
- **Food Search Problem** — collecting all food using heuristic optimization
- Custom heuristics based on Manhattan distance and maze distance caching

### Multi-Agent Search
Implemented in `multiAgents.py`:

- **Minimax Agent** — models adversarial ghost behavior
- **Alpha-Beta Pruning** — optimizes the Minimax tree exploration
- **Expectimax Agent** — handles stochastic ghost behavior

---

## 🛠️ Technologies & Concepts

- Python
- State-space search
- Heuristic design
- Adversarial search & game trees
- Multi-agent systems

---

## 🚀 Getting Started

### Prerequisites

```bash
python --version  # Python 3 required
```

### Run the game

```bash
# Example: run with BFS agent
python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs

# Example: run with A* agent
python pacman.py -l bigMaze -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic

# Example: run Minimax agent
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=3
```

---

## 📁 Project Structure

```
pac-man/
├── src/
│   ├── search.py          # DFS, BFS, UCS, A* implementations
│   ├── searchAgents.py    # Search-based Pac-Man agents
│   ├── multiAgents.py     # Minimax, Alpha-Beta, Expectimax agents
│   └── ...                # Game engine (provided by course)
└── README.md
```

---

## 📌 What I Implemented

- All search algorithms (DFS, BFS, UCS, A*)
- State representations and successor functions
- Custom heuristics for food and corners problems
- Minimax, Alpha-Beta Pruning, and Expectimax agents
- Evaluation functions for intelligent decision making

> The game engine, visualization, and environment setup were provided by the course framework.

---

## 🎓 Academic Context

| | |
|---|---|
| **Course** | Artificial Intelligence |
| **Year** | 3rd year, Bachelor's degree |
| **University** | Technical University |
| **Purpose** | Educational — applying AI algorithms in a game-based environment |

---

## 💡 What I Learned

Working on this project gave me hands-on experience with:

- How search algorithms behave differently in terms of completeness, optimality, and efficiency
- The importance of heuristic design in A* — a bad heuristic can make it no better than BFS
- How Minimax and Alpha-Beta model real decision-making under adversarial conditions
- Thinking in terms of **states, transitions, and goals** rather than procedural logic

---

## 👩‍💻 Author

**Oana** — [GitHub Profile](https://github.com/oanabye)
