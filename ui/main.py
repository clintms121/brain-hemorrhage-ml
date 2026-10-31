import tkinter as tk
from datetime import date
from tkinter import ttk

# Choices match the values in data/brain_hemorrhage_dataset.csv.
GENDERS = ["Female", "Male"]
REGIONS = [
    "Central Asia",
    "Eastern Europe",
    "North America",
    "Oceania",
    "South Asia",
    "Southeast Asia",
    "Sub-Saharan Africa",
    "Western Europe",
]
HEMORRHAGE_TYPES = [
    "Epidural Hematoma",
    "Intracerebral Hemorrhage (ICH)",
    "Intraventricular Hemorrhage",
    "Subarachnoid Hemorrhage (SAH)",
    "Subdural Hematoma",
]
RISK_FACTORS = [
    "Alcohol Consumption",
    "Anticoagulant Use",
    "Diabetes",
    "Head Trauma",
    "Hypertension",
    "Smoking",
]
SYMPTOMS = [
    "Balance Issues",
    "Loss of Consciousness",
    "Nausea/Vomiting",
    "Numbness/Weakness",
    "Seizures",
    "Severe Headache",
    "Slurred Speech",
    "Stiff Neck",
    "Vision Loss",
]
# Best to worst, the same order as OUTCOME_CODES in data/cleaning/process.py.
OUTCOMES = ["Recovered", "Rehabilitation Required", "Severe Disability", "Fatal"]
UNKNOWN_RISK = "None/Unknown"
MIN_AGE, MAX_AGE = 0, 120
# Earliest year in the dataset, so the model has never seen anything before it.
MIN_YEAR = 2000

BACKGROUND = "#f4f6f8"
CARD = "#ffffff"
BORDER = "#d8dee4"
TEXT = "#1f2328"
MUTED = "#59636e"
ACCENT = "#1f6feb"
ACCENT_DARK = "#1858c4"
ERROR = "#cf222e"
FONT = "Segoe UI"


def predict(patient):
    """ Connect model here """
    return None


class ClassifierApp:
    def __init__(self, root):
        self.root = root
        root.title("Brain Hemorrhage Outcome Classifier")
        root.configure(background=BACKGROUND)
        setup_styles(root)

        # Form values
        self.age = tk.StringVar()
        self.gender = tk.StringVar()
        self.region = tk.StringVar()
        self.year = tk.StringVar(value=str(date.today().year))
        self.hemorrhage_type = tk.StringVar()
        self.risk_vars = {}
        self.symptom_vars = {}

        self.build_header()
        body = ttk.Frame(root, style="App.TFrame", padding=(20, 0, 20, 20))
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)
        self.build_form(body).grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        self.build_result_panel(body).grid(row=0, column=1, sticky="nsew")

        self.show_empty_result()
        root.bind("<Return>", lambda event: self.classify())

        # Open at the size the contents need, and don't allow shrinking below it.
        root.update_idletasks()
        root.minsize(root.winfo_reqwidth(), root.winfo_reqheight())

    # ---------- Layout ----------

    def build_header(self):
        header = ttk.Frame(self.root, style="App.TFrame", padding=(20, 18, 20, 14))
        header.pack(fill="x")
        ttk.Label(header, text="Brain Hemorrhage Outcome Classifier", style="Title.TLabel").pack(anchor="w")

    def build_form(self, parent):
        card = make_card(parent)
        form = ttk.Frame(card, style="Card.TFrame", padding=20)
        form.pack(fill="both", expand=True)

        add_section_title(form, "Patient")
        fields = ttk.Frame(form, style="Card.TFrame")
        fields.pack(fill="x")
        fields.columnconfigure(0, weight=1, uniform="field")
        fields.columnconfigure(1, weight=1, uniform="field")

        add_field(fields, 0, 0, "Age (years)", ttk.Entry(fields, textvariable=self.age))

        gender_buttons = ttk.Frame(fields, style="Card.TFrame")
        for gender in GENDERS:
            ttk.Radiobutton(
                gender_buttons, text=gender, value=gender, variable=self.gender, style="Card.TRadiobutton"
            ).pack(side="left", padx=(0, 16))
        add_field(fields, 0, 1, "Gender", gender_buttons)

        region_box = ttk.Combobox(fields, textvariable=self.region, values=REGIONS, state="readonly")
        add_field(fields, 1, 0, "Region", region_box)
        year_box = ttk.Spinbox(fields, textvariable=self.year, from_=MIN_YEAR, to=date.today().year)
        add_field(fields, 1, 1, "Year", year_box)

        add_section_title(form, "Hemorrhage type")
        ttk.Combobox(
            form, textvariable=self.hemorrhage_type, values=HEMORRHAGE_TYPES, state="readonly"
        ).pack(fill="x", pady=(0, 10))

        add_section_title(form, "Medical history (risk factors)", hint="Leave all unchecked if none or unknown.")
        self.risk_vars = add_checkbox_grid(form, RISK_FACTORS)

        add_section_title(form, "Symptoms", hint="Select at least one.")
        self.symptom_vars = add_checkbox_grid(form, SYMPTOMS)

        buttons = ttk.Frame(form, style="Card.TFrame")
        buttons.pack(fill="x", pady=(10, 0))
        ttk.Button(buttons, text="Classify", style="Primary.TButton", command=self.classify).pack(side="right")
        ttk.Button(buttons, text="Clear", style="Secondary.TButton", command=self.clear_form).pack(
            side="right", padx=(0, 8)
        )
        return card

    def build_result_panel(self, parent):
        card = make_card(parent)
        panel = ttk.Frame(card, style="Card.TFrame", padding=20)
        panel.pack(fill="both", expand=True)

        # Wrap widths are kept under the width of the probability rows,
        # so long text wraps instead of making the window wider.
        ttk.Label(panel, text="Predicted outcome", style="Section.TLabel").pack(anchor="w")
        self.result_label = ttk.Label(panel, text="", style="Result.TLabel", wraplength=300)
        self.result_label.pack(anchor="w", pady=(6, 2))
        self.result_note = ttk.Label(panel, text="", style="Hint.TLabel", wraplength=300, justify="left")
        self.result_note.pack(anchor="w")

        ttk.Separator(panel).pack(fill="x", pady=14)
        ttk.Label(panel, text="Probability by outcome", style="Section.TLabel").pack(anchor="w", pady=(0, 8))

        rows = ttk.Frame(panel, style="Card.TFrame")
        rows.pack(fill="x")
        rows.columnconfigure(1, weight=1)
        self.bars = {}
        self.percent_labels = {}
        for row, outcome in enumerate(OUTCOMES):
            ttk.Label(rows, text=outcome, style="Card.TLabel").grid(row=row, column=0, sticky="w", pady=4)
            bar = ttk.Progressbar(rows, style="Outcome.Horizontal.TProgressbar", maximum=100, length=120)
            bar.grid(row=row, column=1, sticky="ew", padx=10)
            percent = ttk.Label(rows, text="", style="Hint.TLabel", width=5, anchor="e")
            percent.grid(row=row, column=2, sticky="e")
            self.bars[outcome] = bar
            self.percent_labels[outcome] = percent

        ttk.Separator(panel).pack(fill="x", pady=14)
        ttk.Label(panel, text="Inputs used", style="Section.TLabel").pack(anchor="w", pady=(0, 6))
        self.summary_frame = ttk.Frame(panel, style="Card.TFrame")
        self.summary_frame.pack(fill="x")
        return card

    # ---------- Actions ----------

    def classify(self):
        patient, errors = self.read_form()
        if errors:
            self.show_form_errors(errors)
            return

        probabilities = predict(patient)
        if probabilities is None:
            self.show_not_connected(patient)
        else:
            self.show_prediction(patient, probabilities)

    def clear_form(self):
        self.age.set("")
        self.gender.set("")
        self.region.set("")
        self.year.set(str(date.today().year))
        self.hemorrhage_type.set("")
        for variable in self.risk_vars.values():
            variable.set(False)
        for variable in self.symptom_vars.values():
            variable.set(False)
        self.show_empty_result()

    def read_form(self):
        """Check the form and return (patient, errors). patient is None if anything is wrong."""
        errors = []

        age = None
        age_text = self.age.get().strip()
        if not age_text.isdigit():
            errors.append("Age must be a whole number.")
        else:
            age = int(age_text)
            if age < MIN_AGE or age > MAX_AGE:
                errors.append(f"Age must be between {MIN_AGE} and {MAX_AGE}.")

        if not self.gender.get():
            errors.append("Choose a gender.")
        if not self.region.get():
            errors.append("Choose a region.")

        year = None
        year_text = self.year.get().strip()
        this_year = date.today().year
        if not year_text.isdigit():
            errors.append(f"Year must be a whole number, e.g. {this_year}.")
        else:
            year = int(year_text)
            if year < MIN_YEAR or year > this_year:
                errors.append(f"Year must be between {MIN_YEAR} and {this_year}.")

        if not self.hemorrhage_type.get():
            errors.append("Choose a hemorrhage type.")

        risk_factors = checked_names(self.risk_vars)
        if not risk_factors:
            risk_factors = [UNKNOWN_RISK]

        symptoms = checked_names(self.symptom_vars)
        if not symptoms:
            errors.append("Select at least one symptom.")

        if errors:
            return None, errors

        patient = {
            "age": age,
            "gender": self.gender.get(),
            "region": self.region.get(),
            "year": year,
            "hemorrhage_type": self.hemorrhage_type.get(),
            "risk_factors": risk_factors,
            "symptoms": symptoms,
        }
        return patient, []

    # ---------- Result panel states ----------

    def show_empty_result(self):
        self.result_label.configure(text="—", style="ResultMuted.TLabel")
        self.result_note.configure(text="Fill in the form and press Classify.", style="Hint.TLabel")
        self.reset_bars()
        self.show_summary(None)

    def show_form_errors(self, errors):
        lines = []
        for error in errors:
            lines.append(f"•  {error}")
        self.result_label.configure(text="Check the form", style="ResultError.TLabel")
        self.result_note.configure(text="\n".join(lines), style="Card.TLabel")
        self.reset_bars()
        self.show_summary(None)

    def show_not_connected(self, patient):
        self.result_label.configure(text="Not available", style="ResultMuted.TLabel")
        self.result_note.configure(
            text="The prediction model isn't connected yet. The form is working, "
            "and these are the inputs it will send.",
            style="Hint.TLabel",
        )
        self.reset_bars()
        self.show_summary(patient)

    def show_prediction(self, patient, probabilities):
        best_outcome = None
        best_probability = -1
        for outcome in OUTCOMES:
            probability = probabilities.get(outcome, 0)
            self.bars[outcome]["value"] = probability * 100
            self.percent_labels[outcome].configure(text=f"{probability:.0%}")
            if probability > best_probability:
                best_outcome = outcome
                best_probability = probability

        self.result_label.configure(text=best_outcome, style="Result.TLabel")
        self.result_note.configure(
            text=f"Most likely outcome ({best_probability:.0%}). Use alongside clinical judgment.",
            style="Hint.TLabel",
        )
        self.show_summary(patient)

    def reset_bars(self):
        for outcome in OUTCOMES:
            self.bars[outcome]["value"] = 0
            self.percent_labels[outcome].configure(text="—")

    def show_summary(self, patient):
        for child in self.summary_frame.winfo_children():
            child.destroy()

        if patient is None:
            ttk.Label(self.summary_frame, text="No patient classified yet.", style="Hint.TLabel").pack(anchor="w")
            return

        rows = [
            ("Age", str(patient["age"])),
            ("Gender", patient["gender"]),
            ("Region", patient["region"]),
            ("Year", str(patient["year"])),
            ("Type", patient["hemorrhage_type"]),
            ("History", ", ".join(patient["risk_factors"])),
            ("Symptoms", ", ".join(patient["symptoms"])),
        ]
        for row, (name, value) in enumerate(rows):
            ttk.Label(self.summary_frame, text=name, style="Field.TLabel").grid(
                row=row, column=0, sticky="nw", padx=(0, 12), pady=1
            )
            ttk.Label(self.summary_frame, text=value, style="Card.TLabel", wraplength=210, justify="left").grid(
                row=row, column=1, sticky="w", pady=1
            )


# ---------- Small layout helpers ----------


def make_card(parent):
    """A white panel with a thin grey border."""
    return tk.Frame(parent, background=CARD, highlightthickness=1, highlightbackground=BORDER, highlightcolor=BORDER)


def add_section_title(parent, title, hint=None):
    heading = ttk.Frame(parent, style="Card.TFrame")
    heading.pack(fill="x", pady=(4, 6))
    ttk.Label(heading, text=title, style="Section.TLabel").pack(side="left")
    if hint:
        ttk.Label(heading, text=hint, style="Hint.TLabel").pack(side="left", padx=(10, 0))


def add_field(parent, row, column, label, widget):
    """Put a small label with its input underneath into a two-column grid."""
    if column == 0:
        padx = (0, 8)
    else:
        padx = (8, 0)
    ttk.Label(parent, text=label, style="Field.TLabel").grid(row=row * 2, column=column, sticky="w", padx=padx)
    widget.grid(row=row * 2 + 1, column=column, sticky="ew", padx=padx, pady=(2, 10))


def add_checkbox_grid(parent, names, columns=3):
    """Lay out one checkbox per name and return {name: BooleanVar}."""
    grid = ttk.Frame(parent, style="Card.TFrame")
    grid.pack(fill="x", pady=(0, 10))
    for column in range(columns):
        grid.columnconfigure(column, weight=1, uniform="checkbox")

    variables = {}
    for index, name in enumerate(names):
        variable = tk.BooleanVar()
        checkbox = ttk.Checkbutton(grid, text=name, variable=variable, style="Card.TCheckbutton")
        checkbox.grid(row=index // columns, column=index % columns, sticky="w", padx=(0, 16), pady=3)
        variables[name] = variable
    return variables


def checked_names(variables):
    """Return the names whose checkbox is ticked."""
    names = []
    for name, variable in variables.items():
        if variable.get():
            names.append(name)
    return names


def setup_styles(root):
    style = ttk.Style(root)
    # "clam" lets us set colors on every widget; the native Windows theme ignores most of them.
    style.theme_use("clam")
    root.option_add("*TCombobox*Listbox.font", (FONT, 10))

    style.configure(".", font=(FONT, 10), background=BACKGROUND, foreground=TEXT)
    style.configure("App.TFrame", background=BACKGROUND)
    style.configure("Card.TFrame", background=CARD)

    style.configure("Title.TLabel", background=BACKGROUND, font=(FONT, 18, "bold"))
    style.configure("Subtitle.TLabel", background=BACKGROUND, foreground=MUTED)
    style.configure("Card.TLabel", background=CARD)
    style.configure("Section.TLabel", background=CARD, font=(FONT, 11, "bold"))
    style.configure("Field.TLabel", background=CARD, foreground=MUTED, font=(FONT, 9))
    style.configure("Hint.TLabel", background=CARD, foreground=MUTED, font=(FONT, 9))
    style.configure("Result.TLabel", background=CARD, foreground=TEXT, font=(FONT, 20, "bold"))
    style.configure("ResultMuted.TLabel", background=CARD, foreground=MUTED, font=(FONT, 20, "bold"))
    style.configure("ResultError.TLabel", background=CARD, foreground=ERROR, font=(FONT, 20, "bold"))

    style.configure("Card.TCheckbutton", background=CARD)
    style.map("Card.TCheckbutton", background=[("active", CARD)])
    style.configure("Card.TRadiobutton", background=CARD)
    style.map("Card.TRadiobutton", background=[("active", CARD)])

    style.configure("TEntry", padding=6, fieldbackground=CARD, bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER)
    style.map("TEntry", bordercolor=[("focus", ACCENT)], lightcolor=[("focus", ACCENT)])
    style.configure(
        "TCombobox", padding=5, fieldbackground=CARD, background=CARD,
        bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER, arrowcolor=MUTED,
    )
    # Without this, a read-only combobox shows its chosen value highlighted in blue.
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", CARD)],
        selectbackground=[("readonly", CARD)],
        selectforeground=[("readonly", TEXT)],
        bordercolor=[("focus", ACCENT)],
    )
    style.configure(
        "TSpinbox", padding=5, fieldbackground=CARD, background=CARD,
        bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER, arrowcolor=MUTED,
    )
    style.map("TSpinbox", bordercolor=[("focus", ACCENT)], lightcolor=[("focus", ACCENT)])

    style.configure(
        "Primary.TButton", background=ACCENT, foreground="white", font=(FONT, 10, "bold"),
        padding=(20, 8), borderwidth=0, focuscolor=ACCENT,
    )
    style.map("Primary.TButton", background=[("pressed", ACCENT_DARK), ("active", ACCENT_DARK)])
    style.configure(
        "Secondary.TButton", background=CARD, foreground=TEXT, padding=(20, 8),
        bordercolor=BORDER, lightcolor=CARD, darkcolor=CARD, focuscolor=CARD,
    )
    style.map("Secondary.TButton", background=[("pressed", BACKGROUND), ("active", BACKGROUND)])

    style.configure(
        "Outcome.Horizontal.TProgressbar", thickness=10, troughcolor="#eaeef2", background=ACCENT,
        bordercolor="#eaeef2", lightcolor=ACCENT, darkcolor=ACCENT,
    )


def enable_sharp_text_on_windows():
    """Without this, Windows stretches the window on high-resolution screens and the text looks blurry."""
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (ImportError, AttributeError, OSError):
        pass


def main():
    enable_sharp_text_on_windows()
    root = tk.Tk()
    ClassifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
