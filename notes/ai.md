Don't take below seriously -- just scratch pad notes.

Parallel solver that communicates globally whenever solutions are found.

- Analytical / Exact Methods
  - Direct algebraic solution: Solve equations symbolically using algebra/calculus. (e.g., Solve $ax + b = 0 \Rightarrow x = -\frac{b}{a}$).
  - Closed-form derivation: Express solution as formula in terms of inputs. (e.g., Solve quadratic: $x = \frac{-b \pm \sqrt{b^{2} - 4ac}}{2a}$).
  - Proof-based reasoning: Show that solution must satisfy certain properties → deduce result. (e.g., Prove optimality by convexity + derivative = 0).
  - Transform methods: Convert to easier domain, solve, invert back. (e.g., Fourier transform PDE, solve in frequency domain).

- Numerical / Approximation Methods
  - Iterative solvers: Improve guess step by step until convergence. (e.g., Newton–Raphson for root-finding: $x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}$).
  - Linear algebra methods: Reduce to matrix equations $Ax = b$. (e.g., LU/QR/SVD factorization to solve $x = A^{-1}b$).
  - Optimization algorithms: Minimize cost function $J(\theta)$. (e.g., Gradient descent: $\theta_{k+1} = \theta_k - \eta \nabla J(\theta_k)$).
  - Simulation: Monte-Carlo / numeric integration to approximate solution. (e.g., Estimate $\pi$ by sampling points in a square and counting inside circle).

- Heuristic / Rule-Based Methods
  - Greedy algorithms: Choose locally optimal step each time. (e.g., Huffman coding, Kruskal MST).
  - Divide-and-conquer: Break into subproblems, solve recursively, combine. (e.g., Merge sort, FFT).
  - Search & enumeration: Systematically explore possibilities. (e.g., Depth-first search, branch-and-bound).
  - Metaheuristics: Guided random search. (e.g., Genetic algorithms, simulated annealing).

- Experimental / Empirical Methods
  - Experimentation: Try candidate solutions, measure outcome. (e.g., A/B testing to pick best policy).
  - Data-driven modeling: Fit statistical or ML model to observed data. (e.g., Linear regression, neural networks).
  - Simulation & what-if analysis: Explore system under different conditions. (e.g., Monte-Carlo for risk analysis).