from tqdm.auto import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from src.models.causal_tft.tft_baseline import TFTBaseline
from src.models.ct import CT


def forecast_tft_values(model: TFTBaseline, dataloader: DataLoader, max_seq_length: int):
    if hasattr(model, "proj_len"): # TODO: remove this part
        projection_length = model.proj_len  
    elif hasattr(model, "projection_length"):
        projection_length = model.projection_length  
    else:
        projection_length = model.projection_horizon
    # format y_true values
    y_true = np.zeros((len(dataloader.dataset), projection_length, 1))
    for i, tau in enumerate(dataloader.dataset.data["future_past_split"]):
        tau = int(tau)
        if not isinstance(model, CT):
            y_true[i] = dataloader.dataset.data["outputs"][i,tau-1:tau+projection_length-1]
        else:
            y_true[i] = torch.tensor(dataloader.dataset.data["outputs"][i, tau:tau+projection_length])
    
    if not isinstance(model, CT):
        # compute predictions from model
        predictions = []
        for batch in tqdm(dataloader):
            predicted_outputs = []
            for key in [
                    "vitals", 
                    "static_features", 
                    "current_treatments", 
                    "outputs", 
                    "prev_treatments", 
                    "prev_outputs", 
                    "active_entries"
                ]:
                    batch[key] = batch[key].to(model.device)
            outputs_scaled = model.forecast(batch).cpu()

            for i in range(batch['vitals'].shape[0]):
                split = int(batch['future_past_split'][i])
                if (split+projection_length < (max_seq_length+1)):
                    predicted_outputs.append(outputs_scaled[i, split :split+projection_length, :])
            predicted_outputs = torch.stack(predicted_outputs)
            predictions.append(predicted_outputs)
        predictions = torch.concat(predictions).numpy()
    else:
        print(f"HORIZON {model.hparams.dataset.projection_horizon}")
        predictions = model.get_autoregressive_predictions(dataloader.dataset)

    return predictions, y_true