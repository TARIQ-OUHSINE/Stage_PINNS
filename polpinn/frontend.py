import sys, threading
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path

from polpinn.PINN import run
from polpinn.save import affichage, cercle_for_frontend
from polpinn.utils import get_output_dir, get_data_dir

params_pinns = {
    "nb_hidden_layer": 2,
    "nb_hidden_perceptron": 32,
    "lr": 0.001,
    "lr_R": 0.00025,
    "epoch": 100000,
    "var_R": True,
}
params = {
    "name": "",
    "D_f": 5 * 10 ** (-16),
    "D_j": 5 * 10 ** (-16),
    "T_1": 20,
    "T_B": 3,
    "P0_f": 1,
    "P0_j": 200,
    "R_bis": 0,
    "def_t": 20,
    "rayon_initialisation": 100 * 10 ** (-9),
}


params_name = {
    "D_f": "Df (m.s-2)",
    "D_j": "Dj (m.s-2) (not necessary)",
    "T_1": "T1 (s)",
    "T_B": "TB (s) (not necessary)",
    "P0_f": "P0_f",
    "P0_j": "P0_j",
    "R_bis": "Solvent radius (not necessary)",
    "def_t": "Temporal definition",
    "rayon_initialisation": "Radius initialization",
    "name": "Folder name",
}

params_pinns_name = {
    "nb_hidden_layer": "Number of hidden layers",
    "nb_hidden_perceptron": "Number of perceptrons per hidden layer",
    "lr": "Learning rate",
    "lr_R": "Radius learning rate",
    "epoch": "Iteration",
    "var_R": "Variable radius",
}


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Polpinn")
        self.window_width = 1000
        self.window_height = 700

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=5)
        self.root.grid_rowconfigure(2, weight=20)
        self.center_window(self.root, self.window_width, self.window_height)

        self.options = ["run", "print", "cercle"]
        self.combo = ttk.Combobox(root, values=self.options)
        self.combo.grid(row=0, column=0)

        self.combo.bind("<<ComboboxSelected>>", self.selection_changed)

        self.frame = tk.Frame(self.root, borderwidth=2, relief="groove")
        self.frame.grid(row=1)

        self.frame2 = tk.Frame(self.root, borderwidth=2, relief="groove")
        self.frame2.grid(row=2)

        self.canvas = None
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def center_window(self, window, width, height):
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()

        x = (screen_width - width) // 2
        y = (screen_height - height) // 2

        window.geometry(f"{width}x{height}+{x}+{y}")

    def selection_changed(self, event):
        self.selected_option = self.combo.get()

        self.frame.destroy()
        self.frame2.destroy()

        self.frame = tk.Frame(self.root, borderwidth=2, relief="groove")
        self.frame.grid(row=1)

        self.frame2 = tk.Frame(self.root, borderwidth=2, relief="groove")
        self.frame2.grid(row=2)

        if self.selected_option == "run":
            self.params_run()

        elif self.selected_option == "print":
            self.chemin_acces_output(show_output_names=True)
            execute_button = tk.Button(
                self.frame, text="Execute", command=self.executer_for_affichage
            )
            execute_button.grid(row=3, column=1, columnspan=2, pady=10)

        elif self.selected_option == "cercle":
            self.chemin_acces_output(show_output_names=True)
            execute_button = tk.Button(
                self.frame, text="Execute", command=self.executer_for_cercle
            )
            execute_button.grid(row=3, column=1, columnspan=2, pady=10)

    def chemin_acces_dataset(self):
        self.dataset_path_label = tk.Label(self.frame, text="Dataset path:")
        self.dataset_path_label.grid(row=1, column=0, padx=10, pady=10)

        self.dataset_path_entry = tk.Entry(self.frame, width=50)
        self.dataset_path_entry.grid(row=1, column=1, padx=10, pady=10)
        self.dataset_path_entry.insert(0, str(get_data_dir()))

        dataset_path = self.dataset_path_entry.get()
        self.dataset_path_button = tk.Button(
            self.frame,
            text="Browse",
            command=lambda: self.browse(
                entry=self.dataset_path_entry,
                update=self.update_datasets,
                initialdir=dataset_path,
            ),
        )
        self.dataset_path_button.grid(row=1, column=2, padx=10, pady=10)

        self.dataset_label = tk.Label(self.frame, text="Dataset:")
        self.dataset_label.grid(row=3, column=0, padx=10, pady=10)
        dataset_variable = tk.StringVar()
        dataset_entry = tk.ttk.Combobox(self.frame, textvariable=dataset_variable)
        dataset_entry.grid(row=3, column=1, padx=10, pady=10)
        self.dataset_entry = dataset_entry
        self.dataset_variable = dataset_variable
        self.update_datasets()

    def update_datasets(self):
        dataset_path = self.dataset_path_entry.get()
        dataset_list = tuple(x.name for x in Path(dataset_path).glob("*") if x.is_dir())
        self.dataset_entry["values"] = dataset_list

    def chemin_acces_output(self, show_output_names, row=1):
        self.output_path_label = tk.Label(self.frame, text="Output access path:")
        self.output_path_label.grid(row=row, column=0, padx=10, pady=10)

        self.output_path_entry = tk.Entry(self.frame, width=50)
        self.output_path_entry.grid(row=row, column=1, padx=10, pady=10)
        self.output_path_entry.insert(0, str(get_output_dir()))

        self.output_path_button = tk.Button(
            self.frame,
            text="Browse",
            command=lambda: self.browse(
                entry=self.output_path_entry,
                update=self.update_output_names,
                initialdir=self.output_path_entry.get(),
            ),
        )
        self.output_path_button.grid(row=row, column=2, padx=10, pady=10)

        if show_output_names:
            self.output_name_label = tk.Label(self.frame, text="Output name:")
            self.output_name_label.grid(row=row + 1, column=0, padx=10, pady=10)
            output_name_variable = tk.StringVar()
            output_name_entry = tk.ttk.Combobox(
                self.frame, textvariable=output_name_variable
            )
            output_name_entry.grid(row=row + 1, column=1, padx=10, pady=10)
            self.output_name_entry = output_name_entry
            self.output_name_variable = output_name_variable
            self.update_output_names()

    def update_output_names(self):
        output_path = self.output_path_entry.get()
        output_names_list = tuple(
            x.name for x in Path(output_path).glob("*") if x.is_dir()
        )
        self.output_name_entry["values"] = output_names_list

    def browse(self, entry, update, initialdir=None):
        path = filedialog.askdirectory(initialdir=initialdir)
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)
            if update is not None:
                update()

    def params_run(self):
        self.chemin_acces_dataset()
        self.chemin_acces_output(row=2, show_output_names=False)
        self.params_pinns_entries = {}
        self.params_pinns_vars = {}
        row_offset = 4
        for i, (key, value) in enumerate(params_pinns.items()):
            label = tk.Label(self.frame, text=f"{params_pinns_name[key]}:")
            label.grid(row=i + row_offset, column=0, padx=10, pady=5)

            if key == "var_R":
                entry_var = tk.BooleanVar()
                entry = tk.ttk.Checkbutton(self.frame, variable=entry_var)
                entry_var.set(True)
            else:
                entry = tk.Entry(self.frame)
                entry_var = None
                if (
                    key == "nb_hidden_layer"
                    or key == "nb_hidden_perceptron"
                    or key == "epoch"
                ):
                    entry.insert(0, str(value))

                else:
                    entry.insert(0, str(float(format(value, f".{6}e"))))
            entry.grid(row=i + row_offset, column=1, padx=10, pady=5)
            self.params_pinns_entries[key] = entry
            self.params_pinns_vars[key] = entry_var

        self.params_entries = {}
        row_offset -= 1
        for i, (key, value) in enumerate(params.items()):
            label = tk.Label(self.frame, text=f"{params_name[key]}:")
            label.grid(row=i + row_offset, column=2, padx=10, pady=5)
            entry = tk.Entry(self.frame)
            if key == "def_t":
                entry.insert(0, str(int(value)))
            elif key == "name":
                entry.insert(0, str(value))
            else:
                entry.insert(0, str(float(format(value, f".{6}e"))))
            entry.grid(row=i + row_offset, column=3, padx=10, pady=5)
            self.params_entries[key] = entry

        execute_button = tk.Button(
            self.frame, text="Execute", command=self.executer_for_run
        )
        execute_button.grid(row=i + 3, column=1, columnspan=2, pady=10)

    def executer_for_run(self):
        path = self.dataset_path_entry.get()
        if not path:
            messagebox.showerror("Error", "Please enter an access path.")
            return
        dataset = self.dataset_variable.get()
        self.frame2.destroy()
        self.frame2 = tk.Frame(self.root, borderwidth=2, relief="groove")
        self.frame2.grid(row=2)
        new_params_pinns = {}
        for key, entry in self.params_pinns_entries.items():
            if key == "var_R":
                entry_var = self.params_pinns_vars[key]
                value = entry_var.get()
                new_params_pinns[key] = value
                continue
            value = entry.get()
            try:
                if (
                    key == "nb_hidden_layer"
                    or key == "nb_hidden_perceptron"
                    or key == "epoch"
                ):
                    value = int(value)
                else:
                    value = float(value)
            except ValueError:
                messagebox.showerror(
                    "Error", f"Invalid value for {params_pinns_name[key]}"
                )
                return
            new_params_pinns[key] = value

        new_params = {}
        for key, entry in self.params_entries.items():
            value = entry.get()
            try:
                if key == "name":
                    value = str(value)
                elif key == "def_t":
                    value = int(value)
                else:
                    value = float(value)
            except ValueError:
                messagebox.showerror(
                    "Error", f"Invalid value for {params_pinns_name[key]}"
                )
                return
            new_params[key] = value

        self.progress_bar = ttk.Progressbar(
            self.frame2, orient="horizontal", length=250, mode="determinate"
        )
        self.progress_bar.grid(column=1, columnspan=2, pady=10)

        self.progress_label = tk.Label(
            self.frame2, text="Progression: 0/" + str(new_params_pinns["epoch"])
        )
        self.progress_label.grid(column=1, columnspan=2, pady=10)

        self.best_R_label = tk.Label(
            self.frame2,
            text="Radius: " + str(new_params["rayon_initialisation"]) + "m",
        )
        self.best_R_label.grid(column=1, columnspan=2, pady=10)

        self.best_loss_label = tk.Label(self.frame2, text="cost: " + str(0))
        self.best_loss_label.grid(column=1, columnspan=2, pady=10)

        max_value = new_params_pinns["epoch"]
        self.progress_bar["maximum"] = max_value

        def update_progress(value, best_R, best_loss):
            self.progress_bar["value"] = value
            self.progress_label.config(text=f"Progression: {value}/{max_value}")
            self.best_R_label.config(
                text=f"Radius: " + str(format(best_R, f".3e")) + " m"
            )
            self.best_loss_label.config(text=f"cost: " + str(format(best_loss, f".2e")))
            self.frame.update_idletasks()

        chemin_corrige = Path(path) / dataset
        output_path = Path(self.output_path_entry.get()) / new_params["name"]
        threading.Thread(
            target=self.run_in_thread,
            args=(
                new_params_pinns,
                new_params,
                chemin_corrige,
                output_path,
                update_progress,
            ),
        ).start()

    def run_in_thread(self, params_pinns, params, path, output_path, update_progress):
        error = run(
            params_pinns=params_pinns,
            params=params,
            data_path=path,
            output_path=output_path,
            update_progress=update_progress,
            no_gui=False,
        )

        if error is not None:
            messagebox.showinfo("Information", error)

    def executer_for_affichage(self):
        self.output_path = (
            Path(self.output_path_entry.get()) / self.output_name_variable.get()
        )
        # if not self.path:
        #     messagebox.showerror("Error", "Please enter an access path.")
        #     return
        # chemin_corrige = self.path.replace("/", "\\")
        error = affichage(str(self.output_path))
        if error is not None:
            messagebox.showinfo("Information", error)

        else:

            # if os.path.isfile(Path(self.path) / "Graphiques" / "loss_and_R.png"):
            #     list_fichier = ["loss_and_R.png"]
            # else:
            #     list_fichier = ["loss.png"]

            # list_fichier.append("S_f_and_S_j.png")
            # list_fichier.append("Polarisation_Ponctuelle.png")
            list_fichier = [
                f.name for f in (Path(self.output_path) / "Graphiques").glob("*.png")
            ]

            self.combo_graph = ttk.Combobox(self.frame, values=list_fichier)
            self.combo_graph.grid(
                row=3,
                padx=10,
                pady=10,
            )

            self.combo_graph.bind("<<ComboboxSelected>>", self.selection_graph)

    def selection_graph(self, event):
        self.frame2.destroy()
        self.frame2 = tk.Frame(self.root, borderwidth=2, relief="groove")
        self.frame2.grid(row=2)
        graph = self.combo_graph.get()
        path_for_graph = Path(self.output_path) / "Graphiques" / graph

        img = Image.open(path_for_graph)
        img = img.resize(
            (self.window_width - 50, self.window_height - 185), Image.LANCZOS
        )

        img_tk = ImageTk.PhotoImage(img)

        label_image = tk.Label(self.frame2)
        label_image.grid(
            row=4,
            column=1,
            padx=10,
            pady=10,
        )
        label_image.config(image=img_tk)
        label_image.image = img_tk

    def executer_for_cercle(self):
        path = Path(self.output_path_entry.get()) / self.output_name_variable.get()
        if not path:
            messagebox.showerror("Error", "Please enter an access path.")
            return

        self.frame2.destroy()
        self.frame2 = tk.Frame(self.root, borderwidth=2, relief="groove")
        self.frame2.grid(row=2)

        cercle = cercle_for_frontend(path)
        if isinstance(cercle, str):
            messagebox.showinfo("Information", cercle)
        else:
            fig, ax = cercle[0], cercle[1]
            self.canvas = FigureCanvasTkAgg(fig, master=self.frame2)
            self.canvas_widget = self.canvas.get_tk_widget()
            self.canvas_widget.pack(fill=tk.BOTH, expand=1)

            self.canvas.draw()

            self.canvas_widget.configure(width=1000, height=600)
            fig.set_size_inches(1000 / fig.dpi, 600 / fig.dpi)
            self.canvas.draw()

    def on_closing(self):
        sys.exit()
