# Active Learning Experiments

Dit project bevat een experimentele omgeving voor het evalueren van verschillende Active Learning (AL) strategieën. De codebase is modulair opgezet om eenvoudig experimenten te draaien met meerdere strategieën, modellen en datasets.

## Structuur

De experimentele setup is opgebouwd rond een aantal kerncomponenten:

### 1. **Strategieën**
Gedefinieerd in de `strategies/` map. Iedere strategie is een klasse die een `run()` methode implementeert en instaat is om:
- Te trainen op een gelabelde subset
- Onzekerheden of informatiewaarden te berekenen op een ongelabelde subset
- Nieuwe datapunten te selecteren voor labeling

### 2. **Modellen**
Machine Learning modellen worden gedefinieerd in `models/`. Momenteel wordt voornamelijk gewerkt met een `ResNet18` model.

### 3. **Datasets**
Datasets worden opgehaald via `datasets/`, waarbij gebruik wordt gemaakt van PyTorch Datasets.

### 4. **Experiment Runner**
Het bestand `run_experiment.py` voert meerdere experimenten parallel uit. Hierin is ondersteuning voor:
- Herhaalde runs per strategie
- Parallelle uitvoering (multiprocessing met `spawn`)
- Dynamische labeling tot de ongelabelde pool uitgeput raakt
- MLflow integratie voor logging van metrics en parameters

### 5. **Logging & Evaluatie**
Experimentele gegevens worden opgeslagen in MLflow:
- Metrics per iteratie (zoals accuracy, training loss, tijd)
- Totaalverbruikte tijd per run
- Visualisaties en artefacten zoals nauwkeurigheidscurves

Analyse gebeurt via een Jupyter-notebook (`active_learning_report.ipynb`), waarin automatisch de MLflow SQLite database wordt uitgelezen.

