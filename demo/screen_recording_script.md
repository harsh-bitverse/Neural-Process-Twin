# Neural Process Twin — Screen Recording & Public Demonstration Script

**Format:** 3-Minute Video Showcase / Presentation Screencast  
**Resolution:** 1080p (1920x1080) at 60 FPS  
**Target Audience:** Scientific Computing Researchers, SciML Engineers, Industrial Thermal Designers

---

## Storyboard & Script Breakdown

### Scene 1: Introduction & Physical Context (0:00 – 0:30)
- **Visual:** Open browser to `http://localhost:8000` showing the Neural Process Twin React Dashboard. Show problem configuration panel with thermal conductivity $k(x, y)$ preview and Gaussian hotspot $q(x, y)$.
- **Voiceover / Caption:**  
  *"Welcome to Neural Process Twin, an open-source scientific computing platform for two-dimensional steady-state heat conduction. In thermal engineering, numerical solvers are accurate but slow, while neural surrogates are fast but uncertified. Neural Process Twin bridges this divide with an explicit dual-mode scientific architecture."*

---

### Scene 2: Verified Numerical Mode (FVM) (0:30 – 1:00)
- **Visual:** Select **Verified Numerical Mode (Conservative FVM)**. Click **Run Simulation**. Show live calculation completion in ~15 ms (or ~300 ms on higher grids). Show green "Certified Numerical Ground Truth" badge. Scroll to diagnostics panel showing:
  - Relative residual: $3.54 \times 10^{-15}$
  - Relative heat flux imbalance: $5.73 \times 10^{-12}$
- **Voiceover / Caption:**  
  *"Our reference pathway is a conservative cell-centered Finite Volume Method solver with harmonic interface conductivities and SuperLU sparse direct factorization. It achieves machine-precision relative algebraic residuals and rigorous second-order spatial convergence under the Method of Manufactured Solutions. This is our trusted engineering ground truth."*

---

### Scene 3: Experimental Neural Surrogate & Physics Smoothing (1:00 – 1:45)
- **Visual:** Toggle to **Experimental Neural Mode (A3 RefinedFNO)**. Highlight amber warning banner: *"Experimental Neural Mode is an uncertified surrogate approximation."* Toggle on **A4 Selective Boundary Jacobi**. Click **Run Simulation**.
- **Visual:** The temperature field renders instantly (<30 ms). Show the latency reduction.
- **Voiceover / Caption:**  
  *"For real-time parameter sweeps, we introduce an experimental multi-resolution Fourier Neural Operator coupled with a local 3-layer convolutional residual block and a Selective Boundary Jacobi smoother. It delivers up to an 8.76x end-to-end API speedup at 256x256 resolution."*

---

### Scene 4: Dual-Mode Comparison & Error Attribution (1:45 – 2:20)
- **Visual:** Switch to **Research Comparison Mode**. Show side-by-side comparison:
  - Left: Verified Numerical FVM
  - Center: Experimental Neural Surrogate
  - Right: Difference Field $\Delta T(x, y)$
- **Visual:** Hover cursor over domain boundary and central hotspot. Show cell-level tooltip displaying FVM temperature, Neural temperature, and local error.
- **Voiceover / Caption:**  
  *"Our comparison mode transparently overlays the ground-truth numerical solution against the neural prediction. Notice how error concentrates along the Dirichlet boundary due to Fourier high-frequency truncation, where the Selective Boundary Jacobi smoother specifically reduces the outer 2-cell error."*

---

### Scene 5: Scientific Governance & Conclusion (2:20 – 3:00)
- **Visual:** Switch to terminal or documentation view showing `reports/NEURAL_PROCESS_TWIN_FINAL_RESEARCH_REPORT.md` and `fig10_empirical_uncertainty_gating_failure.png` / `fig11_rigorous_error_bounds_dilemma.png`.
- **Voiceover / Caption:**  
  *"Crucially, our research demonstrates that empirical uncertainty gating collapses under physical distribution shifts, and rigorous PDE bounds suffer from an O(h^-2) constant explosion. Therefore, Neural Process Twin refuses to promote neural models into the certified pathway. We preserve verified numerical simulation for certified decisions, while empowering engineers with fast neural exploration. All 318 tests, benchmarks, and multi-stage Docker packaging are fully open-source and reproducible on GitHub."*
