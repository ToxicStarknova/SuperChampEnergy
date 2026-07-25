import os
import sys
import customtkinter as ctk

# Ensure workspace root is in python module path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.app import HomeBatteryCalculatorApp

# Global CustomTkinter Configuration
ctk.set_appearance_mode("System")  # Inherits OS Light/Dark theme automatically
ctk.set_default_color_theme("blue")

if __name__ == "__main__":
    root = ctk.CTk()
    app = HomeBatteryCalculatorApp(root)
    root.mainloop()