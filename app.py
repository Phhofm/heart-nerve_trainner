"""Double-click entry point for the desktop app.

    python app.py        # launches the GUI

(In a packaged build this is the target of the Start-Menu shortcut / AppImage.)
"""

import sys

from hnt.ui import run

if __name__ == "__main__":
    sys.exit(run())
