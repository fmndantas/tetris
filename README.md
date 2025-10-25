# Tetris

The classic **Tetris** game implemented as a **CLI** tool in Python.

This project demonstrates the creation of a functional game with board logic and state manipulation. It also served as the starting point for my studies in **Test-Driven Development (TDD)**.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Testing:** TDD (Initial Learning Phase)

---

## 💡 Technical Evolution (Postnotes)

Developed early in my journey, this project reflects an older phase of my coding practices. While functional, the design exhibits high coupling, which provided me with a **valuable lesson on the cost of refactoring** and the importance of **Design Patterns**.

---

## Installation and Execution (Linux)

Due to the reliance on the `keyboard` package on Linux, both installation and execution require root privileges.

### Installation

```bash
git clone [https://github.com/fmndantas/tetris.git](https://github.com/fmndantas/tetris.git)
cd tetris
sudo pip3 install .
```

### Running

Run the game with sudo tetris. Control is done via the keyboard:
- a: Move left
- d: Move right
- s: Move down (soft drop)
- r: Rotate current shape
