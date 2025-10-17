import deepchem as dc
import matplotlib.pyplot as plt
import pandas as pd
from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
from openpom.models.mpnn_pom import MPNNPOMModel
import numpy as np

if __name__ == "__main__":
    TASKS = [
    'alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal',
    'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy',
    'bergamot', 'berry', 'bitter', 'black currant', 'brandy', 'burnt',
    'buttery', 'cabbage', 'camphoreous', 'caramellic', 'cedar', 'celery',
    'chamomile', 'cheesy', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean',
    'clove', 'cocoa', 'coconut', 'coffee', 'cognac', 'cooked', 'cooling',
    'cortex', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy',
    'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruit skin',
    'fruity', 'garlic', 'gassy', 'geranium', 'grape', 'grapefruit', 'grassy',
    'green', 'hawthorn', 'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth',
    'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender', 'leafy', 'leathery',
    'lemon', 'lily', 'malty', 'meaty', 'medicinal', 'melon', 'metallic',
    'milky', 'mint', 'muguet', 'mushroom', 'musk', 'musty', 'natural', 'nutty',
    'odorless', 'oily', 'onion', 'orange', 'orangeflower', 'orris', 'ozone',
    'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn',
    'potato', 'powdery', 'pungent', 'radish', 'raspberry', 'ripe', 'roasted',
    'rose', 'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy',
    'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweaty', 'sweet',
    'tea', 'terpenic', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable',
    'vetiver', 'violet', 'warm', 'waxy', 'weedy', 'winey', 'woody'
    ]

    print("No of tasks: ", len(TASKS))
    n_tasks = len(TASKS)

    n_models = 10

    models_list = []
    for i in range(n_models):
        model = MPNNPOMModel(n_tasks = n_tasks,
                                batch_size=128,
                                class_imbalance_ratio = None,
                                loss_aggr_type = 'sum',
                                node_out_feats = 100,
                                edge_hidden_feats = 75,
                                edge_out_feats = 100,
                                num_step_message_passing = 5,
                                mpnn_residual = True,
                                message_aggregator_type = 'sum',
                                mode = 'classification',
                                number_atom_features = GraphConvConstants.ATOM_FDIM,
                                number_bond_features = GraphConvConstants.BOND_FDIM,
                                n_classes = 1,
                                readout_type = 'set2set',
                                num_step_set2set = 3,
                                num_layer_set2set = 2,
                                ffn_hidden_list= [392, 392],
                                ffn_embeddings = 256,
                                ffn_activation = 'relu',
                                ffn_dropout_p = 0.12,
                                ffn_dropout_at_input_no_act = False,
                                weight_decay = 1e-5,
                                self_loop = False,
                                optimizer_name = 'adam',
                                log_frequency = 32,
                                model_dir = f'./models/ensemble_models/experiments_{i+1}',
                                device_name='cuda')
        model.restore(f"./models/ensemble_models/experiments_{i+1}/checkpoint2.pt")
        models_list.append(model)


    def predict_odors(models_list, smiles):
        featurizer = GraphFeaturizer()
        featurized_data = featurizer.featurize(smiles)
        preds = []
        for model in models_list:   
            prediction = model.predict(dc.data.NumpyDataset(featurized_data))
            preds.append(prediction)
        preds_arr = np.asarray(preds)
        ensemble_preds = np.mean(preds_arr, axis=0)
        return ensemble_preds


    def plot_predictions(predictions):
        # Visualize the top 10 predictions
        prediction_df = pd.DataFrame({'odors': TASKS, 'prediction': predictions.squeeze()}).sort_values(by='prediction', ascending=False)
        prediction_df[:10].plot.bar(x='odors', y='prediction')


    import tkinter as tk
    from tkinter import ttk, messagebox
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    from PIL import Image, ImageTk
    import io
    import json

    # Import RDKit for SMILES rendering
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        RDKitAvailable = True
    except ImportError:
        RDKitAvailable = False

    class OdorPredictorApp:
        FONT_LARGE = ("Arial", 15, "bold")
        FONT_REGULAR = ("Arial", 13)
        FONT_LABEL = ("Arial", 12, "bold")
        FONT_ENTRY = ("Arial", 13)
        FONT_BUTTON = ("Arial", 13, "bold")
        FONT_SMALL = ("Arial", 11)

        def __init__(self, root):
            self.root = root
            root.title("Odor Predictor")

            # Give user option to maximize the window
            self.root.geometry("990x660")
            self.root.minsize(700, 500)
            self.root.resizable(True, True)
            try:
                self.root.update_idletasks()
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()
                size = tuple(int(dim) for dim in self.root.geometry().split("+")[0].split("x"))
                x = (screen_width - size[0]) // 2
                y = (screen_height - size[1]) // 2
                self.root.geometry(f"{size[0]}x{size[1]}+{x}+{y}")
            except Exception:
                pass

            # Main frame setup
            self.mainframe = ttk.Frame(root, padding="18 18 18 18")
            self.mainframe.pack(fill=tk.BOTH, expand=True)
            self.mainframe.pack_propagate(False)

            # -- Input frame at top --
            self.input_frame = ttk.Frame(self.mainframe)
            self.input_frame.pack(side=tk.TOP, fill=tk.X, anchor='n', pady=(6, 0))

            self.smiles_label = ttk.Label(self.input_frame, text="Enter SMILES string (only one):", font=self.FONT_LABEL)
            self.smiles_label.grid(row=0, column=0, sticky=tk.W, pady=2)
            self.smiles_entry = ttk.Entry(self.input_frame, width=65, font=self.FONT_ENTRY)
            self.smiles_entry.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5, padx=(0, 2))
            self.smiles_entry.bind('<Return>', lambda event: self.predict())
            self.smiles_entry.bind('<KeyRelease>', self.on_smiles_entry_change)

            # Structure image for SMILES (realtime)
            self.smiles_img_panel = ttk.Label(self.input_frame)
            self.smiles_img_panel.grid(row=1, column=1, padx=24, pady=2, sticky=tk.W)
            self._structure_photo = None  # Reference to avoid gc

            self.topn_label = ttk.Label(self.input_frame, text="Top N to display (max 138):", font=self.FONT_LABEL)
            self.topn_label.grid(row=2, column=0, sticky=tk.W, pady=2)
            self.topn_entry = ttk.Entry(self.input_frame, width=10, font=self.FONT_ENTRY)
            self.topn_entry.insert(0, "10")
            self.topn_entry.grid(row=2, column=0, sticky=tk.W, padx=(224, 0), pady=2)

            self.button_frame = ttk.Frame(self.input_frame)
            self.button_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=9)
            self.predict_button = ttk.Button(self.button_frame, text="Predict Odors", command=self.predict)
            self.predict_button.grid(row=0, column=0, padx=0, sticky=tk.W)
            self.clear_button = ttk.Button(self.button_frame, text="Clear", command=self.clear_output)
            self.clear_button.grid(row=0, column=1, padx=8, sticky=tk.W)

            # -- New: Copy TXT and JSON buttons --
            self.copy_buttons_frame = ttk.Frame(self.input_frame)
            self.copy_buttons_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=1)
            self.copy_txt_button = ttk.Button(self.copy_buttons_frame, text="Copy Results (TXT)", command=self.copy_to_clipboard_txt)
            self.copy_txt_button.grid(row=0, column=0, padx=(0, 5))
            self.copy_json_button = ttk.Button(self.copy_buttons_frame, text="Copy Results (JSON)", command=self.copy_to_clipboard_json)
            self.copy_json_button.grid(row=0, column=1, padx=(0, 5))
            self.copy_txt_button.configure(style="TButton")
            self.copy_json_button.configure(style="TButton")
            # Variable to cache topN prediction DataFrame after prediction
            self.last_topn_df = None

            # Increase button font
            self.predict_button.configure(style="TButton")
            self.clear_button.configure(style="TButton")

            # -- Output area with canvas and both scrollbars --
            self.output_area_frame = ttk.Frame(self.mainframe)
            self.output_area_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(18,3))

            self.output_canvas = tk.Canvas(self.output_area_frame, borderwidth=0, highlightthickness=0, bg='white')
            self.output_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            self.output_vscroll = ttk.Scrollbar(self.output_area_frame, orient="vertical", command=self.output_canvas.yview)
            self.output_vscroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.output_hscroll = ttk.Scrollbar(self.mainframe, orient="horizontal", command=self.output_canvas.xview)
            self.output_hscroll.pack(side=tk.BOTTOM, fill=tk.X, anchor='s')

            self.output_canvas.configure(yscrollcommand=self.output_vscroll.set, xscrollcommand=self.output_hscroll.set)
            
            self.output_frame = ttk.Frame(self.output_canvas)
            self.output_frame_id = self.output_canvas.create_window((0,0), window=self.output_frame, anchor='nw')

            self.output_frame.bind("<Configure>", self._on_output_frame_configure)
            self.output_canvas.bind("<Configure>", self._on_canvas_configure)
            self.root.bind("<Configure>", self._on_root_configure)
            self.canvas_list = []

            # Set style (fonts, buttons)
            style = ttk.Style()
            style.configure("TButton", font=self.FONT_BUTTON, padding=6)
            style.configure("TLabel", font=self.FONT_REGULAR, padding=2)
            style.configure("Treeview.Heading", font=self.FONT_LABEL)

            self._bind_mousewheel()

        def _bind_mousewheel(self):
            def _on_mousewheel(event):
                if event.state & 1:  # Shift is pressed
                    self.output_canvas.xview_scroll(int(-1*(event.delta/120)), "units")
                else:
                    self.output_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            self.output_canvas.bind_all('<MouseWheel>', _on_mousewheel)

        def _on_output_frame_configure(self, event):
            self.output_canvas.configure(scrollregion=self.output_canvas.bbox("all"))

        def _on_canvas_configure(self, event):
            pass

        def _on_root_configure(self, event):
            if event.widget == self.root:
                self.output_canvas.config(width=self.output_area_frame.winfo_width(), height=self.output_area_frame.winfo_height())

        def clear_output(self):
            for widget in self.output_frame.winfo_children():
                widget.destroy()
            self.canvas_list.clear()
            self.smiles_entry.delete(0, tk.END)
            self.topn_entry.delete(0, tk.END)
            self.topn_entry.insert(0, "10")
            self.output_canvas.yview_moveto(0)
            self.output_canvas.xview_moveto(0)
            self.smiles_img_panel.configure(image="", text="")  # Clear structure image
            self.last_topn_df = None

        def on_smiles_entry_change(self, event=None):
            smiles = self.smiles_entry.get().strip()
            if not smiles or not RDKitAvailable:
                self.smiles_img_panel.configure(image="", text="")
                self._structure_photo = None
                return
            # Only draw if it's not multiple smiles
            if "," in smiles:
                self.smiles_img_panel.configure(image="", text="(Only one SMILES allowed)", font=self.FONT_SMALL)
                self._structure_photo = None
                return
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                try:
                    img = Draw.MolToImage(mol, size=(120,120), kekulize=True)
                    bio = io.BytesIO()
                    img.save(bio, format="PNG")
                    bio.seek(0)
                    pil_img = Image.open(bio)
                    self._structure_photo = ImageTk.PhotoImage(pil_img)
                    self.smiles_img_panel.configure(image=self._structure_photo, text="")
                except Exception:
                    self.smiles_img_panel.configure(image="", text="(Could not render)", font=self.FONT_SMALL)
                    self._structure_photo = None
            else:
                self.smiles_img_panel.configure(image="", text="(Invalid SMILES)", font=self.FONT_SMALL)
                self._structure_photo = None

        def predict(self):
            for widget in self.output_frame.winfo_children():
                widget.destroy()
            self.canvas_list.clear()
            self.output_canvas.yview_moveto(0)
            self.output_canvas.xview_moveto(0)
            self.last_topn_df = None

            smiles_str = self.smiles_entry.get().strip()
            # Only allow one SMILES string, error if more than one
            if not smiles_str:
                messagebox.showerror("Error", "No SMILES entered.")
                return
            if ',' in smiles_str:
                messagebox.showerror("Error", "Please enter only one SMILES string (no commas).")
                return

            # Validate SMILES with RDKit if available
            if RDKitAvailable:
                mol = Chem.MolFromSmiles(smiles_str)
                if mol is None:
                    messagebox.showerror("Error", "Invalid SMILES string.")
                    return

            smiles = [smiles_str]

            try:
                # Validate top N (must be <=138)
                top_n_str = self.topn_entry.get()
                try:
                    top_n = int(top_n_str)
                    if top_n <= 0:
                        raise ValueError
                    if top_n > 138:
                        messagebox.showwarning("Warning", "Top N too large, limiting to 138.")
                        top_n = 138
                except Exception:
                    messagebox.showwarning("Warning", "Invalid top N entered. Using 10.")
                    top_n = 10

                predictions = predict_odors(models_list, smiles)
                predictions_arr = predictions.squeeze()
                if predictions_arr.ndim == 1:
                    predictions_arr = predictions_arr.reshape(1, -1)

                # Should only be one row now
                max_img_width = 0
                for i, p in enumerate(predictions_arr):
                    prediction_df = pd.DataFrame({'odors': TASKS, 'prediction': p})
                    prediction_df = prediction_df.sort_values(by='prediction', ascending=False)
                    top_df = prediction_df[:top_n]
                    # Cache for copy buttons
                    self.last_topn_df = top_df.copy()

                    plot_width = max(7, int(top_n*1.15))
                    # Use Arial fonts for readability
                    fig, ax = plt.subplots(figsize=(plot_width, 6))
                    bars = ax.bar(top_df['odors'], top_df['prediction'], color="#97bae8", width=0.5)
                    ax.set_xticks(range(len(top_df['odors'])))
                    ax.set_xticklabels(
                        top_df['odors'],
                        rotation=38,
                        ha='right',
                        fontsize=15,
                        fontweight='medium',
                        fontname="Arial"
                    )
                    # ax.margins(x=0.19)
                    plt.title(
                        f"Top {top_n} Odor Predictions\nSMILES: {smiles[0]}",
                        fontsize=18,
                        fontweight="bold",
                        fontname="Arial"
                    )
                    plt.ylabel('Prediction', fontsize=17, fontname="Arial")
                    ax.tick_params(axis='y', labelsize=14)
                    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=7))
                    plt.setp(ax.get_yticklabels(), rotation=90, fontsize=13, fontname="Arial")

                    # Calculate y-limit extension based on the annotations
                    top_heights = [bar.get_height() for bar in bars]
                    if top_heights:
                        max_height = max(top_heights)
                        # Allow +10% "buffer" above the max bar for annotation room
                        y_max = max_height * 1.12  # gives enough room for values above bar
                        ax.set_ylim(0, y_max)

                    for bar in bars:
                        height = bar.get_height()
                        # Place annotation slightly below the upper bound if close to top
                        y_offset = 4
                        if top_heights:
                            y_max_plot = ax.get_ylim()[1]
                            if height > y_max_plot * 0.97:
                                y_offset = -10
                        ax.annotate(
                            f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, y_offset),
                            textcoords="offset points",
                            ha='center',
                            va='bottom' if y_offset >= 0 else 'top',
                            fontsize=13,
                            weight="semibold",
                            color="navy",
                            rotation=0,
                            fontname="Arial",
                            clip_on=True
                        )

                    plt.tight_layout()

                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight')
                    plt.close(fig)
                    buf.seek(0)
                    img = Image.open(buf)
                    img_width, img_height = img.size

                    if img_width > max_img_width:
                        max_img_width = img_width

                    photo = ImageTk.PhotoImage(img)
                    img_label = ttk.Label(self.output_frame, image=photo)
                    img_label.image = photo
                    img_label.pack(pady=7, anchor='w')
                    self.canvas_list.append(img_label)

                frame_width = max(max_img_width + 45, self.output_canvas.winfo_width())
                total_height = sum(label.winfo_reqheight() for label in self.canvas_list) + 21*len(self.canvas_list)
                if total_height < 1:
                    total_height = 10
                self.output_canvas.config(scrollregion=(0, 0, frame_width, total_height))
            except Exception as e:
                messagebox.showerror("Error", f"Prediction failed:\n{e}")

        def copy_to_clipboard_txt(self):
            """
            Copy topN table to clipboard in TXT format: "Odor\tPrediction" per line.
            """
            if self.last_topn_df is None or self.last_topn_df.empty:
                messagebox.showwarning("Warning", "No predictions to copy. Please run a prediction first.")
                return
            txt_lines = ["Odor\tPrediction"]
            for _, row in self.last_topn_df.iterrows():
                txt_lines.append(f"{row['odors']}\t{row['prediction']:.5f}")
            text_to_copy = "\n".join(txt_lines)
            self.root.clipboard_clear()
            self.root.clipboard_append(text_to_copy)
            self.root.update()  # Needed for clipboard
            messagebox.showinfo("Copied", "Top N results copied to clipboard as TXT.")

        def copy_to_clipboard_json(self):
            """
            Copy topN table to clipboard in JSON format: list of dicts with odor and prediction.
            """
            if self.last_topn_df is None or self.last_topn_df.empty:
                messagebox.showwarning("Warning", "No predictions to copy. Please run a prediction first.")
                return
            data = [
                {"odor": row["odors"], "prediction": float(row["prediction"])}
                for _, row in self.last_topn_df.iterrows()
            ]
            json_txt = json.dumps(data, indent=2)
            self.root.clipboard_clear()
            self.root.clipboard_append(json_txt)
            self.root.update()  # Needed for clipboard
            messagebox.showinfo("Copied", "Top N results copied to clipboard as JSON.")


    def run_smiles_ui():
        root = tk.Tk()
        app = OdorPredictorApp(root)
        root.mainloop()

    run_smiles_ui()
