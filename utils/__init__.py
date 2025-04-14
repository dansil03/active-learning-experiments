'''
Doel: 
Het script `tracking.py` bevat hulpfuncties voor het **loggen van parameters en metrics naar MLflow** tijdens het trainen van machine learning modellen.

Het stelt je in staat om:
- Parameters zoals `modelnaam`, `strategie`, `initial_size`, etc. automatisch te loggen met `log_params()`.
- Prestaties zoals `accuracy`, `loss`, `tijd per iteratie`, etc. te loggen met `log_metrics()`.

Deze informatie wordt centraal opgeslagen in MLflow, zodat je experimenten kunt:
- Terugvinden
- Vergelijken
- Visualiseren (in de MLflow UI)
'''

