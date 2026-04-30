# online_sleep_score

Python GUI for online sleep-state scoring from Intan-style recordings.

## Input

Select a recording folder containing:

- `amplifier.dat`
- `amplifier.xml`

## Run

```powershell
python run_sleep_score_gui.py
```

## Workflow

1. Choose `Basepath`.
2. Optionally choose `Previous datapath`.
3. Press `Estimate parameters from previous session` to fill thresholds from a previous recording.
4. Adjust thresholds if needed.
5. Press `Run scoring`.

If previous parameters are not estimated, the app estimates channels from the first `Estimation min` minutes of the current recording.

## Output

Results are saved in the recording folder:

```text
<basename>.OnlineSleepState.states.mat
```

The MAT file contains:

- `SleepState`
- `OnlineSleepFeatures`

## Windows App Build

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\build_windows_app.ps1
```

The build output is:

```text
dist\OnlineSleepScore\OnlineSleepScore.exe
```

Distribute the full `dist\OnlineSleepScore` folder, not the `.exe` alone.

