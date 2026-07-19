# Upstair Policy — Review & Tutorial TODO

A learn-by-doing checklist for training the N2 18-DOF humanoid to climb **real stairs (~16 cm rise / ~30 cm tread)**. Each task explains *why*, *where* (file:line), *what to change* (code sketch), and *how to verify*. Implement manually and check items off as you go.

Environment: `conda activate unitree-rl`.

---

## Context — why this work is needed

You flipped the config into stair mode (`curriculum=True`, `terrain_proportions` now 45% up-stairs / 20% down-stairs, `tracking_lin_vel` 1.2→1.4, `default_up_joint_pos` 1.0→1.1) and set `play.py` to evaluate on pure up-stairs. The scaffolding is sound, but two things block a real-stair policy:

1. **The training curriculum never produces the stairs you want to climb.** The stair generator caps step height at **9 cm** (`difficulty*0.10`, difficulty ≤ 0.9) with a fixed **0.4 m** tread. Your own commented `play.py` targets **16 cm** — so you evaluate on steps the policy has never seen.
2. **The deployed actor is blind and memoryless** — single-frame proprioception only (63 obs: commands, base angular vel, gravity, dof pos/vel, last action). No height scan, no memory, not even base linear velocity. The 96-point height scan exists but is fed **only to the critic** (asymmetric actor-critic). Blindly reacting to a 16 cm riser with no temporal context is very hard.

Chosen direction: target 16 cm/30 cm stairs, and add temporal history to the actor. This TODO covers both, plus the reward/curriculum tuning that makes tall-stair climbing learnable.

---

## Primer — how the terrain + observation pipeline fits together

- **Terrain build:** `HumanoidTerrain` (in `humanoid/utils/terrain.py`) is what actually runs (instantiated at `legged_robot.py:261`), *not* the base `Terrain` class. Its `make_terrain` is at `terrain.py:214-248`. A 10×10 grid of tiles: rows = difficulty levels (0→9), columns = terrain types chosen from `terrain_proportions`.
- **Curriculum:** robots start at level 0 (`max_init_terrain_level=0`) and are promoted when they walk `env_length/2 = 2 m` horizontally from their tile origin (`_update_terrain_curriculum`, `legged_robot.py:508-528`). Vertical climb is **not** credited.
- **Observations:** actor obs built in `n2_env.py:146-153` (63 dims); critic/privileged obs in `n2_env.py:156-176` (221 dims, incl. 96 height samples). Frame-stacking is already wired (`legged_robot.py:679-684` builds the history deque; `n2_env.py:185-190` stacks it; `sim2sim.py:111-142` rebuilds it at deploy) but currently **off** (`frame_stack` commented out).
- **Terrain-relative rewards (already correct):** `_reward_base_height` (`legged_robot.py:1055-1058`) and height termination (`legged_robot.py:146`) both subtract `terrain_h`, so the 0.698 height target and 0.4 fall threshold track the step the robot stands on — they do **not** fight the climb. Good design; leave the *mechanism* alone, only tune magnitudes.

---

## TODO 1 — Make the stair curriculum reach 16 cm / 30 cm  ⭐ highest impact

**Why:** This is the single change that decides whether the policy ever trains on your target stairs. Today max rise = 9 cm, tread = 0.4 m.

**Where:** `humanoid/utils/terrain.py:220` and `:240-245` (the `HumanoidTerrain.make_terrain` stair branches).

**What to change:**
- The stair height reuses `discrete_obstacles_height = difficulty * 0.10`. Give stairs their own variable so you don't couple them to the obstacles branch, and size it so the top level ≈ 16 cm. Since curriculum difficulty maxes at `9/10 = 0.9`, use a coefficient of ~`0.18` (0.9 × 0.18 = 16.2 cm):
  ```python
  # terrain.py, inside HumanoidTerrain.make_terrain (~line 220)
  stair_height = difficulty * 0.18      # 0 → 16.2 cm across levels 0..9
  ```
- Tighten the tread from 0.4 m toward a realistic 0.30 m, and use the new variable, in both stair branches (`:241` up, `:244` down):
  ```python
  terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.30, step_height= stair_height, platform_size=1.)   # up
  terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.30, step_height=-stair_height, platform_size=1.)   # down
  ```
- **Sanity on quantization:** heights snap to `vertical_scale = 0.005 m`. 0.162/0.005 = 32.4 → fine. Tread 0.30/`horizontal_scale`(0.1) = 3 cells → fine.

**Learn-more:** the level→difficulty map is `difficulty = i/num_rows` (`terrain.py:99`). If you later want the *top* level to be exactly 16 cm, either raise `num_rows` and rescale, or clamp: `stair_height = min(difficulty, 0.9)*0.178`. Keep level 0 near-flat so the curriculum has an on-ramp.

**Verify:** run `play.py --task=n2` after training a bit and watch the robot on increasing rows; or temporarily set `env_cfg.terrain.num_rows=10, curriculum=False, max_init_terrain_level=9` in `play.py` to spawn directly on the hardest stairs. Alternatively use the `selected=True` block already stubbed in `play.py:44-48` (that path needs the `self.vertical_scale/horizontal_scale` attributes added in `terrain.py:63-64` — those exist precisely to support it).

---

## TODO 2 — Give the actor temporal history (frame-stacking)  ⭐

**Why:** A blind, memoryless MLP cannot infer velocity or "I'm mid-step-up" from one frame. Stacking the last N proprioceptive frames lets it estimate base velocity and terrain interaction — the standard recipe for blind stair climbing. Almost all the plumbing already exists; you're flipping it on and keeping train/deploy in sync.

**Where / what to change (4 edits, all must agree on N):**

1. **`humanoid/envs/n2/n2_config.py:55-71` (class `env`)** — enable stacking and resize obs:
   ```python
   frame_stack = 15                       # N; 10–15 is typical
   num_single_obs = 63
   num_observations = frame_stack * num_single_obs   # 945 for N=15
   ```
   Leave `num_privileged_obs = 221` as-is — the critic stays single-frame (asymmetric is fine).

2. **`sim2sim/configs/n2_18dof.yaml` — `frame_stack` / `num_obs`** — must match training exactly, or the deployed policy sees garbage:
   ```yaml
   frame_stack: 15
   num_obs: 945     # num_single_obs * frame_stack
   ```
   No code change needed in `sim2sim.py` — it already builds the `hist_obs` deque and flattens it (`sim2sim.py:111-142`). Confirm the stacking **order** matches: env appends newest to the right (`n2_env.py:186`) and stacks `range(maxlen)` oldest→newest; sim2sim appends+popleft (`:136-137`) then reads `range(frame_stack)` oldest→newest. They agree — just don't reorder either.

3. **Clear the history on episode reset (latent bug — fix before enabling).** `obs_history` is built once (`legged_robot.py:681-684`) and never cleared per-env on reset, so a freshly reset env keeps the *previous episode's* frames in its stack. Harmless today (stack off) but wrong once N>1. In the reset path (`reset_idx` in `legged_robot.py`), zero the reset envs' rows in every history frame:
   ```python
   # after reset of env_ids, if frame stacking is on:
   if self.cfg.env.frame_stack is not None:
       for i in range(len(self.obs_history)):
           self.obs_history[i][env_ids] *= 0.
   ```
   Verify with a quick print that a just-reset env has an all-zero history tail.

4. **Actor network size:** none needed — `actor_hidden_dims=[256,128]` reads `num_observations` automatically. (Optional: widen the first layer, e.g. `[512,256,128]`, since input grew 63→945.)

**Verify:** start training (`train.py --task=n2 --headless --num_envs 4096`), confirm it builds without shape errors and the actor input dim prints as 945. After a short run, `play.py --task=n2` should still load. Then run `sim2sim.py` — if the robot behaves like in Isaac, stacking is consistent; if it instantly falls, the `frame_stack`/`num_obs`/order don't match.

**Learn-more:** the README notes the project originally had an LSTM locomotion version. Frame-stacking is the simpler, export-friendly (JIT) equivalent of short-horizon memory and plays nicely with the existing sim2sim loop, which is why it's the right first move over reintroducing recurrence.

---

## TODO 3 — Add a swing-foot clearance reward  ⭐

**Why:** The biggest failure mode for *blind* stair climbing is catching the toe on the riser. There is currently **no** reward encouraging the swing foot to lift high enough to clear a step. Adding one is usually the difference between "shuffles into the step and falls" and "steps up cleanly."

**Where:** add a reward method in `humanoid/envs/n2/n2_env.py` (foot state is already tracked: `self.feet_pos`, `self.contacts`), and a scale in `n2_config.py` `class scales`.

**What to add (sketch):** reward swing-foot height above the local terrain up to a target clearance (≈ step height + margin, so ~0.18 m):
```python
def _reward_feet_clearance(self):
    # height of each foot above the terrain under it, while that foot is in swing
    feet_z = self.feet_pos[:, :, 2] - self.terrain_h.unsqueeze(-1)   # terrain-relative
    swing = ~self.contacts                                            # feet not in contact
    target = 0.18
    # reward getting close to target clearance during swing; ignore stance feet
    err = torch.square(torch.clamp(target - feet_z, min=0.0)) * swing
    return torch.exp(-err.sum(dim=1) * 50)
```
Then in `n2_config.py` `class scales`: start small, e.g. `feet_clearance = 1.0`, and tune. Keep it modest so it doesn't dominate `tracking_lin_vel`.

**Verify:** during `play.py`, watch swing-foot apex height; it should rise as you increase the scale. If the robot starts prancing on flat ground, lower the scale or gate the reward on forward command (`self.commands[:,0].abs() > self.min_cmd_vel`).

**Learn-more:** foot clearance is often paired with a small *stumble* penalty on horizontal foot force during swing (there's a `_reward_stumble` stub at `legged_robot.py:1139`). Consider enabling a light version once clearance works.

---

## TODO 4 — Retune base-height penalty for stairs

**Why:** `base_height` scale is **-30.0** with `only_positive_rewards=True`. The mechanism is terrain-relative (good), but `terrain_h` snaps by a full step as the robot crosses each riser (`_get_ground_heights` takes the max tread under the base), injecting a large squared-error spike at every step. At -30 that spike is big relative to the ~1.4 tracking reward, and because `only_positive_rewards` clips the *total* to ≥0, a few large penalties can zero out the whole step's reward — killing the learning signal exactly on the hardest frames.

**Where:** `n2_config.py:264` (`base_height = -30.0`), `n2_config.py:253` (`only_positive_rewards = True`).

**What to try (pick one, measure):**
- Soften the penalty: `base_height = -10.0` (or -5.0). Least invasive.
- Add a deadband so small deviations are free (edit `_reward_base_height`, `legged_robot.py:1055-1058`):
  ```python
  err = torch.abs(base_height - self.cfg.rewards.base_height_target)
  return torch.square(torch.clamp(err - 0.05, min=0.0))   # 5 cm deadband
  ```
- Consider `only_positive_rewards = False` for stair training so penalties still shape behavior instead of being clipped away — but watch for early-training instability; if returns collapse, revert.

**Verify:** log the per-term reward (the runner already writes episode reward breakdowns to `logs/`). Watch `rew_base_height` vs `rew_tracking_lin_vel` on stair tiles — the base-height term should be a correction, not the dominant signal.

---

## TODO 5 — Tame surface roughness on stair tiles

**Why:** Every stair tile gets `add_roughness(np.random.uniform(0.01, 0.05))` (`terrain.py:242,245`) — up to ±5 cm of noise. On low curriculum levels where the step itself is only 1–4 cm, the roughness can exceed the step height and effectively destroy the clean staircase shape, making the early on-ramp chaotic and inconsistent.

**Where:** `terrain.py:242` and `:245`.

**What to try:** scale roughness with difficulty so early levels stay clean, and cap it below the step height:
```python
rough = min(0.02, 0.5 * stair_height)      # never rougher than half a step, ≤2 cm
add_roughness(terrain, np.random.uniform(0.005, max(0.005, rough)))
```
This keeps some texture (good for sim-to-real friction/edge robustness) without swamping shallow steps.

**Verify:** visualize a low-level stair tile (spawn at `max_init_terrain_level=1`, `curriculum=False`) in `play.py` — steps should look like steps, not gravel.

---

## TODO 6 — (Optional) Credit vertical progress in the curriculum

**Why:** Level-up requires **2 m of horizontal displacement** (`_update_terrain_curriculum`, `legged_robot.py:518-520`) and ignores height gained. On up-stairs, forward progress per unit effort is low and climb height is uncredited, so robots advance through stair levels slowly — they may plateau on easy stairs.

**Where:** `legged_robot.py:518-523`.

**What to try (conservative):** blend a fraction of climbed height into the progress metric, or lower the up-threshold for stair columns. Simplest safe version — count vertical rise toward "distance":
```python
horiz = torch.norm(self.root_states[env_ids,:2] - self.env_origins[env_ids,:2], dim=1)
climb = torch.clamp(self.root_states[env_ids,2] - self.env_origins[env_ids,2], min=0.0)
distance = horiz + 2.0 * climb          # weight climb; tune the factor
```
Keep it mild — over-crediting climb lets robots ratchet up levels without actually traversing.

**Verify:** watch mean `terrain_level` over training (logged) rise faster/more steadily than before; ensure it still converges rather than everyone pinning at max.

---

## TODO 7 — Housekeeping (low priority, do anytime)

- **`MUJOCO_LOG.TXT` is tracked in git** and shows as modified every run. Add it to `.gitignore` and `git rm --cached MUJOCO_LOG.TXT`. It's a runtime log, not source.
- **Variable naming:** once TODO 1 is done, the stairs no longer borrow `discrete_obstacles_height`; the new `stair_height` name documents intent. Good.
- **`terrain.py:63-64`** (`self.vertical_scale`, `self.horizontal_scale`) are only read by the `selected_terrain()` path (`terrain.py:114-115`), which runs only when `terrain.selected=True`. Keep them — they're what makes the commented `selected`-mode block in `play.py:44-48` work for manual step-height testing. Not dead, just conditional.

---

## Suggested implementation order & overall verification

1. **TODO 1** (stairs reach 16 cm) — nothing else matters until the target terrain exists.
2. **TODO 5** (roughness) — cheap, makes the early curriculum sane.
3. **TODO 2** (frame-stacking) — the actor capability upgrade; keep train + sim2sim in lockstep.
4. **TODO 3** (foot clearance) — the reward that most directly enables stepping up.
5. **TODO 4** (base-height retune) — remove the penalty that fights the climb.
6. **TODO 6** (curriculum, optional) — only if progression stalls.
7. **TODO 7** (hygiene) — anytime.

**End-to-end check after each milestone:**
- Train headless: `conda activate unitree-rl && python humanoid/scripts/train.py --task=n2 --headless --num_envs 4096`. Confirm no shape errors; watch mean reward and mean `terrain_level` climb in the logged metrics.
- Export + view: `python humanoid/scripts/play.py --task=n2` (exports JIT policy, renders on stairs). Increase `max_init_terrain_level` in `play.py` to test the hardest steps directly.
- Sim2sim reality check: copy the exported policy to `sim2sim/policy/`, update `sim2sim/configs/n2_18dof.yaml` (esp. `frame_stack`/`num_obs` after TODO 2), then `python sim2sim/sim2sim.py`. If Mujoco behavior matches Isaac Gym, your obs pipeline is consistent.

**Watch for these failure signatures:**
- Instant fall in sim2sim but fine in Isaac → `frame_stack`/`num_obs`/scaling mismatch (TODO 2).
- Robot shuffles into risers and trips → clearance reward too weak (TODO 3) or roughness too high (TODO 5).
- Learning stalls / reward flat on stairs → base-height penalty dominating or `only_positive_rewards` clipping (TODO 4).
- Everyone stuck on easy levels → curriculum progression (TODO 6).
