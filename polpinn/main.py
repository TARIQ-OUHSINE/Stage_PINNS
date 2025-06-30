import argparse
import matplotlib

from polpinn.PINN import run
from polpinn.save import reload_model, affichage, cercle
from polpinn.utils import get_output_dir, get_data_dir

params_pinns = {
    "nb_hidden_layer": 2,
    "nb_hidden_perceptron": 32,
    "lr": 0.001,
    "lr_R": 0.0001,
    # "epoch": 20000,
    "epoch": 10000,
    "var_R": False,
}
params = {
    "D_f": 5 * 10 ** (-16),
    "D_j": 5 * 10 ** (-16),
    "T_1": 20,
    "T_B": 1,
    "P0_f": 1,
    "P0_j": 200,
    "R_bis": 0.0,
    "def_t": 20,
    "rayon_initialisation": 2.5 * 10 ** (-7),
    "name": "Sim9",
}

data_path = get_data_dir() / "Moyenne_homo_HyperP"
output_path = get_output_dir() / params["name"]


def main_nogui(no_interaction):
    answer = True
    default_answer = 1
    while answer:
        question = "Menu:\n"
        question += "    1 - Run experiment\n"
        question += "    2 - Reload model\n"
        question += "    3 - Cercle\n"
        question += "    4 - Affichage\n"
        question += "    5 - Exit"
        print(question)
        if no_interaction:
            answer = ""
        else:
            answer = input(f"Your choice [default={default_answer}]:")
        if answer == "":
            answer = default_answer
        answer = int(answer)
        print(f"answer: {answer}")
        if answer == 1:
            run(
                params_pinns,
                params,
                data_path=data_path,
                output_path=output_path,
                no_interaction=no_interaction,
                no_gui=True,
            )
        elif answer == 2:
            model = reload_model(output_path)
            # model()
        elif answer == 3:
            cercle(output_path, no_interaction=no_interaction)
        elif answer == 4:
            affichage(output_path)
        else:
            answer = False

        default_answer = answer + 1


def main_gui():
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    from polpinn.frontend import App
    import tkinter as tk

    parser = argparse.ArgumentParser(description="Main program")
    parser.add_argument(
        "-Y",
        dest="no_interaction",
        action="store_const",
        const=True,
        default=False,
        help="Force answers to yes (no user interaction)",
    )
    parser.add_argument(
        "-no_gui",
        dest="no_gui",
        action="store_const",
        const=True,
        default=False,
        help="Disable graphical user interface (use terminal instead)",
    )
    no_interaction = parser.parse_args().no_interaction
    no_gui = parser.parse_args().no_gui
    if no_gui:
        main_nogui(no_interaction=no_interaction)
    else:
        # matplotlib.use("TkAgg")
        main_gui()
