# pipemania-solver
Artificial Intelligence solution for the Pipemania puzzle.

# PIPEMANIA AI Solver 🧩

A Python project to solve a variation of the classic **Pipemania** puzzle using Artificial Intelligence techniques. The goal is to rotate the pieces on a grid so that all pipes are correctly connected, allowing the water to flow without leaks.

This project explores problem-solving through state modeling and search algorithms, aiming to find the optimal sequence of moves.

---

## 🎯 Project Goals
- Read a PIPEMANIA board from input.
- Use AI search techniques to solve the puzzle by rotating the pieces into the correct positions.
- Output the solved board in a readable format.

---

## 🧠 Approach
The solution is based on classic search algorithms provided as part of the course framework. Several strategies were tested, including:
- **Depth-First Search (DFS)**
- **Breadth-First Search (BFS)**
- **Greedy Search**
- **A\* Search**

After experimenting, **A\*** was selected as the most effective, thanks to its balance between cost and heuristic estimation, making it well-suited for this type of problem.

---

## 🛠️ Main Features
- Board representation with support for different pipe types and orientations.
- Piece rotation logic and valid move generation.
- Heuristic function for informed search algorithms.
- Goal checking and state transition handling.
- Support for testing with various board configurations.

---

<img width="661" alt="image" src="https://github.com/user-attachments/assets/84437452-3518-49af-8577-1fbc4d1747f9" />

---

## ⚙️ How to Run
```bash
python3 pipe.py < initial-state.txt
```

### Example Input:
```
FB VC VD
BC BB LV
FB FB FE
```

### Example Output:
```
FB VB VE
BD BE LV
FC FC FC
```

---

## 🧰 Technologies
- **Python 3.8.2**
- Standard libraries: `sys`, `copy`, `time`
- Optional: `numpy` (for matrix operations)

---
