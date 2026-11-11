; Inno Setup script — BODY Studio (Windows).
; Per-user install (no admin). Ships the app + the trainer repo + a bundled `uv`
; so first run can create an isolated venv and install deps (see bootstrap.py).
;
; Build:  iscc packaging/windows/installer.iss
; Expects PyInstaller output in dist\BodyStudio\ and a vendored `uv.exe`
; + a copy of traiNNer-redux in packaging\vendor\.

#define AppName "BODY Studio"
#define AppVersion "0.1.4"
#define AppExe "BodyStudio.exe"
; Paths in Inno are relative to this script; the repo root is two levels up.
#define RepoDist AddBackslash(SourcePath) + "..\..\dist"

[Setup]
AppName={#AppName}
AppVersion={#AppVersion}
DefaultDirName={localappdata}\BodyStudio
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir={#RepoDist}
OutputBaseFilename=BodyStudio-Setup
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern

[Files]
; The staged PyInstaller folder already contains the app, `trainer/`, `uv.exe`
; and the `hnt` sources — keep them together so find_repo()/uv_path() resolve.
Source: "{#RepoDist}\BodyStudio\*"; DestDir: "{app}\app"; Flags: recursesubdirs

[Icons]
Name: "{userprograms}\{#AppName}"; Filename: "{app}\app\{#AppExe}"

[Run]
Filename: "{app}\app\{#AppExe}"; Description: "Launch {#AppName}"; Flags: postinstall nowait
