; Inno Setup script — models are NOT bundled; the first-run wizard downloads them.
[Setup]
AppName=Dual-Screen Meeting AI Assistant
AppVersion=1.0
DefaultDirName={autopf}\DualScreenAI
DefaultGroupName=Dual-Screen AI
OutputDir=..\dist
OutputBaseFilename=DualScreenAI-Setup
Compression=lzma2
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
PrivilegesRequiredOverridesAllowed=dialog
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\DualScreenAI.exe

[Files]
Source: "..\dist\DualScreenAI\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\Dual-Screen AI"; Filename: "{app}\DualScreenAI.exe"
Name: "{autodesktop}\Dual-Screen AI"; Filename: "{app}\DualScreenAI.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Run]
Filename: "{app}\DualScreenAI.exe"; Description: "Launch Dual-Screen AI"; Flags: nowait postinstall skipifsilent
