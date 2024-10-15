import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline


def plot_feature_importances(data_pipeline: Pipeline, model):
    fea_imp_df = pd.DataFrame(
        {
            "features": data_pipeline.get_feature_names_out(),
            "importances": model.feature_importances_,
        }
    ).sort_values("importances", ascending=True)

    fea_imp_figure = plt.figure(figsize=(20, 8))
    plt.barh(fea_imp_df["features"], fea_imp_df["importances"], color="skyblue")
    plt.xlabel("Importances")
    plt.ylabel("Features")

    return fea_imp_figure
