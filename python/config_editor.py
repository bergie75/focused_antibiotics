import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import yaml
from expanded_model import par_names, full_names, antibiotic_modes, run_simulation, run_sens_test
import os


class YamlEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("YAML Variable Editor")
        self.path = None
        self.data = {}
        self.decrease_by = 0

        self.descriptive_names = full_names
        self.display_to_key = {}

        toolbar = tk.Frame(root)
        toolbar.pack(fill="x", padx=10, pady=10)

        tk.Button(toolbar, text="Open YAML", command=self.open_file).pack(side="left")

        tk.Label(toolbar, text="Variable:").pack(side="left", padx=(15, 5))

        self.variable_choice = tk.StringVar()
        self.dropdown = tk.OptionMenu(toolbar, self.variable_choice, "")
        self.dropdown.pack(side="left")
        self.variable_choice.trace_add("write", self.load_variable)

        tk.Label(toolbar, text="Antibiotic mode:").pack(side="left", padx=(15, 5))
        self.antibiotic_mode_choice = tk.StringVar(value=antibiotic_modes[0] if antibiotic_modes else "")
        self.antibiotic_mode_dropdown = tk.OptionMenu(toolbar, self.antibiotic_mode_choice, *antibiotic_modes)
        self.antibiotic_mode_dropdown.pack(side="left")

        tk.Label(toolbar, text="Decrease by:").pack(side="left", padx=(10, 5))
        self.input_value = tk.StringVar()
        self.value_entry = tk.Entry(toolbar, textvariable=self.input_value, width=12)
        self.value_entry.pack(side="left")
        self.save_value_button = tk.Button(toolbar, text="Save value", command=self.save_dec_by)
        self.save_value_button.pack(side="left", padx=(5, 0))

        self.simulate_button = tk.Button(toolbar, text="Run simulation", command=self.run_sim)
        self.simulate_button.pack(side="left", padx=(5, 0))
        self.simulate_button.config(state="disabled")

        # for sensitivity testing
        self.sens_button = tk.Button(toolbar, text="Run sensitivity analysis", command=self.run_sens)
        self.sens_button.pack(side="left", padx=(5, 0))
        self.sens_button.config(state="disabled")

        tk.Label(root, text="Value:").pack(anchor="w", padx=10)
        self.value_editor = tk.Text(root, height=10, wrap="word")
        self.value_editor.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        value = self.input_value.get()

        tk.Button(root, text="Save", command=self.save_variable).pack(
            anchor="e", padx=10, pady=(0, 10)
        )

    def open_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )
        if not filename:
            return

        try:
            data = yaml.safe_load(Path(filename).read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("The YAML file must contain top-level variables.")
        except (OSError, yaml.YAMLError, ValueError) as error:
            messagebox.showerror("Open failed", str(error))
            return

        self.path = Path(filename)
        self.data = data
        self.update_dropdown()
        self.simulate_button.config(state="normal")
        self.sens_button.config(state="normal")
        self.root.title(f"YAML Variable Editor — {self.path.name}")

    def run_sim(self):
        mode = self.antibiotic_mode_choice.get()
        with open(self.path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            
        pars = [cfg[x] for x in par_names]
        run_simulation(pars, save_results=True, plot_results=True, mode=mode, dec_value=self.decrease_by)

    def run_sens(self):
        mode = self.antibiotic_mode_choice.get()
        with open(self.path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            
        immutable_pars = [cfg[x] for x in par_names]
        run_sens_test(immutable_pars, mode=mode, dec_value=self.decrease_by)
    
    def update_dropdown(self):
        menu = self.dropdown["menu"]
        menu.delete(0, "end")
        self.display_to_key = {}

        keys = list(self.data.keys())
        for key in keys:
            label = self.descriptive_names.get(str(key), str(key))
            self.display_to_key[label] = key
            menu.add_command(
                label=label,
                command=lambda selected=label: self.variable_choice.set(selected)
            )

        if keys:
            first_label = self.descriptive_names.get(str(keys[0]), str(keys[0]))
            self.variable_choice.set(first_label)
        else:
            self.variable_choice.set("")
            self.value_editor.delete("1.0", "end")

    def load_variable(self, *_):
        selected = self.display_to_key.get(self.variable_choice.get())
        if selected not in self.data:
            return

        self.value_editor.delete("1.0", "end")
        self.value_editor.insert("1.0", str(self.data[selected]))

    def save_dec_by(self):
        self.decrease_by = self.input_value.get()

    def save_variable(self):
        selected = self.display_to_key.get(self.variable_choice.get())
        if not self.path or selected not in self.data:
            messagebox.showerror("Save failed", "Open a YAML file and select a variable.")
            return

        text = self.value_editor.get("1.0", "end-1c")

        try:
            # Parse the edited value as YAML, so numbers, booleans, lists,
            # and null values keep their YAML types.
            self.data[selected] = yaml.safe_load(text)
            self.path.write_text(
                yaml.safe_dump(self.data, sort_keys=False, allow_unicode=True),
                encoding="utf-8"
            )
        except (OSError, yaml.YAMLError) as error:
            messagebox.showerror("Save failed", str(error))
            return

        messagebox.showinfo("Saved", f"Updated '{selected}' in {self.path.name}")


root = tk.Tk()
root.geometry("600x350")
YamlEditor(root)
root.mainloop()