### **Report: On-the-Fly Lightcone Generation Protocol**

#### **1. Core Concept: Separation of Dynamics and Observation**

To generate physically accurate lensing maps, the simulation must separate the **Dynamical State** (evolving gravity) from the **Observational State** (what the observer sees).

* **Dynamical State:** The "Real" particles used to compute time steps. They must remain synchronized at the specific snapshot scale factor ().
* **Observational State:** "Ghost" particles created temporarily via interpolation. They exist at specific lightcone crossing times () and are discarded immediately after measurement.

---

#### **2. The Pipeline Algorithm**

This process runs **inside** the main simulation loop (on-the-fly), not on saved snapshots.

**Loop: Evolve from Scale Factor **

1. **Check for Crossing:**
Determine if the lightcone shell (defined by comoving distances  and ) intersects the simulation box volume in this time interval.
2. **Create Ghost Particles (Drift/Interpolation):**
For every particle that crosses the lightcone, we calculate the exact fractional time of crossing.
* *Input:* Particle state at  ().
* *Operation:* Linearly interpolate trajectory to .
* *Output:* Temporary array `x_ghost`.


3. **Apply PGD (Sharpening):**
Apply Potential Gradient Descent forces to `x_ghost`.
* *Why:* This corrects the "fuzziness" of PM halos. It must be done on the ghost particles so the halo is sharp exactly when the light ray passes through it.


4. **Apply Coordinate Transformation (Tiling):**
Transform `x_ghost` into global coordinates `x_global` relative to the observer.
* *Condition A (Local Volume):* If the box is the observer's home box (Distance ), **Do Not Rotate**.
* *Condition B (Distant Volume):* If the box is a copy (Distance ), **Apply Random Rotation**.
* *Shift:* Apply integer box offsets to place the tile at the correct distance.


5. **Lensing Deposit:**
Project `x_global` onto the 2D lens plane grid.
6. **Cleanup:**
Delete `x_ghost` and `x_global`. The simulation resumes using the original synchronized particles at .

---

#### **3. Mathematical & Physical Definitions**

##### **A. The Drift (Interpolation)**

We assume the velocity  is constant over the small timestep. The displacement is proportional to the integral of the equation of motion.

**The Drift Factor :**
Derived from  and :


##### **B. The Rotation (Isotropy)**

To prevent the "Hall of Mirrors" effect where the same structure repeats periodically along the line of sight, we apply a rotation matrix .

* **If  (Observer's Box):**  (Identity Matrix).
* *Reason:* Preserves local environment (e.g., Milky Way orientation).


* **If  (Replica Box):** .
* *Reason:* Breaks periodicity and ensures statistical isotropy.



##### **C. The Shift (Tiling)**

The shift places the box in the correct global position. It is strictly an integer multiple of the box size .

* **Note:** The shift is **not** modulo. It is additive. The "modulo" logic is implicitly handled because the input coordinates  are always inside .

---

#### **4. Summary Table**

| Component | Logic / Formula | Timing |
| --- | --- | --- |
| **Drift** | Interpolate  to  | **Inside Loop** (Before Gravity update) |
| **PGD** | Shift  to sharpen halos | **Inside Loop** (On drifted particles) |
| **Rotation** | **Identity** if <br>

<br>**Random** if  | **Inside Loop** (Before Shift) |
| **Shift** | Add  | **Inside Loop** (Before Deposit) |
| **Simulation** | Continues unaffected | **Post-Loop** (Uses non-drifted particles) |
