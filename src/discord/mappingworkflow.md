A practical workflow is: have the LLM emit valid Quake 3 .map text, run an automated q3map2 compile to .bsp, then launch a scripted ioquake3 client that loads the map, teleports a camera entity/player to a known position/orientation, and calls the engine screenshot command to dump a 2D image. This can all be wrapped in a service that returns the final image path/bytes to whatever orchestrates the LLM.​

Overall architecture
LLM side: takes a high‑level prompt (layout, room graph, style) and outputs .map content following the Quake 3 MAP text format (entities, brushes, textures).​

Tooling side: a pipeline script (Python, etc.) saves that text as generated.map, compiles it with q3map2 to generated.bsp, and, if compilation succeeds, spawns ioquake3 in a “capture” mode to render and screenshot from a camera.​

This separation keeps the LLM responsible only for the map source while deterministic tools handle compilation and rendering.​

Letting the LLM author .map
Provide the model with a concise schema/spec of the MAP format: worldspawn entity with key‑values plus brushes, each brush face as three points and a texture/surface description.​

Constrain generation by:

Supplying a template .map with minimal boilerplate (origin, simple room, a camera marker entity) and asking the model to only fill pre‑marked sections.​

Fixing grid units and allowed textures so q3map2 can always find materials.​

The pipeline should validate syntax and run a fast “lint” (e.g., small script to check entity blocks, face plane counts) before compiling, and fall back to a default test map if validation fails.​

Compiling maps automatically
Install q3map2 (from GtkRadiant or standalone) and call it from your script with the usual passes, pointing it at a controlled game directory.​

A common automated sequence is:

BSP: structural compile with -game q3a and your common switches.​

VIS: run visibility if you care about accurate lighting and performance; for rapid iteration you can skip or simplify.​

LIGHT: compute lightmaps; or use fast options for preview.​

The script should parse q3map2’s exit code/log, and only proceed to rendering if a .bsp was successfully produced in baseq3/maps.​

Getting a controllable camera
There are two main approaches to a camera for snapshots:

Use the player viewpoint:

Instruct the LLM to place info entities (e.g., a custom marker or a specific info_player_start) in the .map with known coordinates and target angles.​

Write a tiny game mod or config script that, on map load, teleports the player to that origin/angles (via server console commands or mod code) and disables HUD and weapon viewmodel.​

Implement a dedicated camera entity:

Create a simple Quake 3 mod that spawns a “camera” entity which the client can switch to, similar to spectator free‑cam or scripted camera systems seen in mods.​

The LLM can then emit entities like target_position/target_relay or a custom camera entity with keys that your mod reads to place the view.​

In both cases, the important part is that the LLM knows how to encode camera position/orientation into entity key‑values in the .map that the runtime will interpret.​

Headless rendering and snapshot output
Use ioquake3 because it has modernized rendering and is scriptable via configs and console commands.​

Launch it from your pipeline with a fixed config:

Auto‑exec a config that sets resolution, field of view, hides HUD and weapon (cg_draw2D 0, cg_drawGun 0 or engine‑specific equivalents) and binds a key/command for screenshotJPEG.​

Use command‑line options to:

Load your mod (if using a custom camera).​

Start a local server on the generated map, execute a script that positions the camera or player, then runs screenshotJPEG once and quits.​

Many Quake 3 engines support scripting screenshot capture and some community scripts already automate clean screenshots without UI clutter.​
Your orchestration code can then watch the baseq3/screenshots directory, grab the latest shotNNNN.jpg, and return it or post‑process (crop/resize) as needed.​