; Inno Setup script — HEART / NERVE Trainer (Windows).
; Per-user install (no admin). Ships the app + the trainer repo + a bundled `uv`
; so first run can create an isolated venv and install deps (see bootstrap.py).
;
; Build:  iscc packaging/windows/installer.iss
; Expects PyInstaller output in dist\HeartNerveTrainer\ and a vendored `uv.exe`
; + a copy of traiNNer-redux in packaging\vendor\.

#define AppName "HEART / NERVE Trainer"
#define AppVersion "0.1.0"
#define AppExe "HeartNerveTrainer.exe"

[Setup]
AppName={#AppName}
AppVersion={#AppVersion}
DefaultDirName={localappdata}\HeartNerveTrainer
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputBaseFilename=HeartNerveTrainer-Setup
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern

[Files]
; The staged PyInstaller folder already contains the app, `trainer/`, `uv.exe`
; and the `hnt` sources — keep them together so find_repo()/uv_path() resolve.
Source: "dist\HeartNerveTrainer\*"; DestDir: "{app}\app"; Flags: recursesubdirs

[Icons]
Name: "{userprograms}\{#AppName}"; Filename: "{app}\app\{#AppExe}"

[Run]
Filename: "{app}\app\{#AppExe}"; Description: "Launch {#AppName}"; Flags: postinstall nowait
