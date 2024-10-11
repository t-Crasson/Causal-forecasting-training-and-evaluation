import warnings
from math import e, log
from typing import Any, Callable, Literal
import joblib
import json

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
import statsmodels.formula.api as smf


def update_mapping(df: pd.DataFrame, dict_mapping: dict, price_column: str, od_id_column: str, time_column: str, min_delta_price: int = 3) -> None:
    """
    This function aims to build a dict mapping each od_id to a list of sequence indexes (min/max) having the same price 
    """
    # this time df contains all day_xs
    od_id_df = df.sort_values(time_column)
    od_id = od_id_df[od_id_column].values[0]
    day_xs = od_id_df[time_column].values.tolist()
    prices = od_id_df[price_column].values.tolist()

    dict_mapping[od_id] = []
    current_price_first_index = 0
    for i in range(len(day_xs)):
        if abs(prices[i] - prices[current_price_first_index]) >= min_delta_price:
            dict_mapping[od_id].append((current_price_first_index, i))
            current_price_first_index = i

    if current_price_first_index != len(day_xs) - 1:
        dict_mapping[od_id].append((current_price_first_index, len(day_xs) - 1))


def _indexes_day_x_both_ways_iterator(
    dict_mapping: dict,
    od_id: str,
) -> Any:
    """
    This functions provide an iterator on sequence indexes providing both left sequence indexes and right sequence indexes
    for each time step with a price change
    """
    if od_id not in dict_mapping:
        yield None
    else:
        for sequence_index, (day_x_min_index, day_x_max_index) in enumerate(dict_mapping[od_id][:-1]):
            next_day_x_min_index, next_day_x_max_index = dict_mapping[od_id][sequence_index + 1]
            if next_day_x_max_index <= next_day_x_min_index + 1 or day_x_max_index <= day_x_min_index + 1:
                """
                Specific case where
                - price changed from p to p' at day_x = next_day_x_min.
                - Price also changed at day_x = day_x_min to become p
                - price changed from p to p' at day_x = next_day_x_min.
                - Price also changed at day_x = day_x_min to become p
                - left: [day_x_min; day_x_max[ --> price = p
                - right: [next_day_x_min=day_x_max; next_day_x_max[ --> price = p'
                - We should only keep data at ]day_x_min; day_x_max[ and ]next_day_x_min;next_day_x_max[
                - We should only keep data at ]day_x_min; day_x_max[ and ]next_day_x_min;next_day_x_max[
                if right length = 1:
                    - For now we skip this case and don't use it
                    - Way to handle it:
                        - We check values right after. If it same price as p. We skip the case
                        - If price is =/= p, we use the new sequence
                if left length = 1:
                    - We do same as the case where right = 1 but in the other direction
                if left length = 1:
                    - We do same as the case where right = 1 but in the other direction
                """
                continue

            # we don't use demand à t where t in the price changed day_x. event when looking at next series
            next_day_x_min_index += 1
            day_x_min_index += 1
            prediction_index = day_x_max_index

            yield (
                sequence_index,
                day_x_min_index,
                day_x_max_index,
                next_day_x_min_index,
                next_day_x_max_index,
                prediction_index,
            )


def tft_model_both_ways_iterator(
    models: list[tuple[str, Any]],
    dict_mapping: dict,
    od_id: str,
    features_df: pd.DataFrame,
    dict_mapping_df: pd.DataFrame,
    day_init: int,
    price_column: str = "reference_price_standard_2",  # different from original price column
) -> Any:
    """
    This iterator uses the _indexes_day_x_both_ways_iterator and an lgbm model to predict the values needed to compute an elasticity value
    at a price changed time step

    features_df has the original data while dict_mapping_df has the filtered data
    """
    features_df.sort_values("day_x", inplace=True)
    features_df.reset_index(inplace=True, drop=True)

    dict_mapping_df.sort_values("day_x", inplace=True)
    dict_mapping_df.reset_index(inplace=True, drop=True)

    filtered_price_values = dict_mapping_df[price_column].values.tolist()
    filtered_day_x_values = dict_mapping_df["day_x"].values.tolist()
    original_day_x_values = features_df["day_x"].values.tolist()

    for model_name, model in models:

        for res_iter in _indexes_day_x_both_ways_iterator(dict_mapping=dict_mapping, od_id=od_id):
            if res_iter is None:
                continue
            (
                sequence_index,
                day_x_min_index,
                day_x_max_index,
                next_day_x_min_index,
                next_day_x_max_index,
                prediction_index,
            ) = res_iter
            # get values associated to the new price
            prediction_day_x = filtered_day_x_values[prediction_index]
            original_prediction_index = original_day_x_values.index(prediction_day_x)
            right_model_day_xs = filtered_day_x_values[next_day_x_min_index:next_day_x_max_index]


            right_fit_predictions = predict_with_tft(tft_model=model, od_id_df=features_df.copy(), day_init=day_init)
            right_fit_predictions = np.array([right_fit_predictions[idx] for idx, o_day_x in enumerate(original_day_x_values) if o_day_x in filtered_day_x_values])
            assert len(right_fit_predictions) == len(filtered_day_x_values)
            
            right_prediction_price = filtered_price_values[next_day_x_min_index]
            right_predicted_demand = right_fit_predictions[prediction_index]
            right_fit_predictions = right_fit_predictions[next_day_x_min_index:next_day_x_max_index]

            # get values associated to previous price
            left_fit_predictions_df = features_df.copy().reset_index()
            left_prediction_price = filtered_price_values[day_x_min_index]
            left_fit_predictions_df.loc[original_prediction_index, price_column] = left_prediction_price

            left_fit_predictions = predict_with_tft(tft_model=model, od_id_df=left_fit_predictions_df, day_init=day_init)
            left_fit_predictions = np.array([left_fit_predictions[idx] for idx, o_day_x in enumerate(original_day_x_values) if o_day_x in filtered_day_x_values])
            assert len(left_fit_predictions) == len(filtered_day_x_values)
            left_predicted_demand = left_fit_predictions[prediction_index]
            left_model_day_xs = filtered_day_x_values[day_x_min_index:day_x_max_index]
            left_fit_predictions = left_fit_predictions[day_x_min_index:day_x_max_index]


            original_left_days = [day_x for day_x in original_day_x_values if filtered_day_x_values[day_x_min_index] <= day_x < prediction_day_x]
            original_right_days = [day_x for day_x in original_day_x_values if prediction_day_x < day_x <= filtered_day_x_values[next_day_x_max_index-1]]

            yield (
                model_name,
                sequence_index,
                prediction_index,
                left_fit_predictions,
                left_prediction_price,
                left_model_day_xs,
                left_predicted_demand,
                right_fit_predictions,
                right_prediction_price,
                right_model_day_xs,
                right_predicted_demand,
                original_left_days,
                original_right_days,
            )


def rdd_both_ways_iterator(
    models: list[tuple[str, Callable]],
    dict_mapping: dict,
    od_id: str,
    day_xs: np.ndarray,
    demand: np.ndarray,
    prices: np.ndarray,
) -> Any:
    """
    This iterator uses the _indexes_day_x_both_ways_iterator and use the RDD process to compute elasticity values associted to a time step with a price changed
    """
    for model_name, model_fit_method in models:
        is_single_model = model_name.startswith("single_")
        for res_iter in _indexes_day_x_both_ways_iterator(dict_mapping=dict_mapping, od_id=od_id):
            if res_iter is None: 
                continue
            (
                sequence_index,
                day_x_min_index,
                day_x_max_index,
                next_day_x_min_index,
                next_day_x_max_index,
                prediction_index,
            ) = res_iter
            if not is_single_model:
                left_model_results = _fit_rdd_model(
                    model_name=model_name,
                    model_fit_method=model_fit_method,
                    day_x_min_index=day_x_min_index,
                    day_x_max_index=day_x_max_index,
                    prediction_index=prediction_index,
                    original_day_xs=day_xs[:],
                    original_demand=demand[:],
                    original_prices=prices[:],
                    prediction_mode="next",
                )

                right_model_results = _fit_rdd_model(
                    model_name=model_name,
                    model_fit_method=model_fit_method,
                    day_x_min_index=next_day_x_min_index,
                    day_x_max_index=next_day_x_max_index,
                    prediction_index=prediction_index,
                    original_day_xs=day_xs[:],
                    original_demand=demand[:],
                    original_prices=prices[:],
                    prediction_mode="prev",
                )
            else:
                (
                    left_model_results,
                    right_model_results
                ) = _fit_both_ways_rdd_model(
                    model_name=model_name,
                    model_fit_method=model_fit_method,
                    day_x_min_index=day_x_min_index,
                    day_x_max_index=day_x_max_index,
                    next_day_x_min_index=next_day_x_min_index,
                    next_day_x_max_index=next_day_x_max_index,
                    prediction_index=prediction_index,
                    original_day_xs=day_xs[:],
                    original_demand=demand[:],
                    original_prices=prices[:],
                )
            yield (
                model_name,
                sequence_index,
                day_x_min_index,
                day_x_max_index,
                next_day_x_min_index,
                next_day_x_max_index,
                prediction_index,
                left_model_results,
                right_model_results,
            )



def _compute_od_id_rdd_values(
    od_id_df: pd.DataFrame,
    rdd_models: list[tuple[str, Callable]],
    demand_column: str,
    price_column: str,
    dict_mapping: dict,
    time_column: str,
    od_id_column: str,
    tqdm_logger: Any = None,
    evaluation_models: list[tuple[str, Any]] = None,
    day_init: int = 20,
    tft_models: list[tuple[str, Any]] = None,
    dict_mapping_query_str: str = None,
    filter_warning: bool = False,
) -> pd.DataFrame:
    """
    Compute all the rdd values of a dataset
    """
    if filter_warning:
        warnings.filterwarnings("ignore")

    dict_mapping_df = od_id_df.copy().sort_values(time_column).reset_index(drop=True)
    original_day_xs = od_id_df[time_column].values.tolist()
    if dict_mapping_query_str is not None:
        dict_mapping_df = dict_mapping_df.query(dict_mapping_query_str)

    od_id = od_id_df[od_id_column].values[0]
    prices = dict_mapping_df[price_column].values
    demand = dict_mapping_df[demand_column].values
    day_xs = dict_mapping_df[time_column].values

    res_dict: dict[str, list] = {
        time_column: [],
        "left_price": [],
        "right_price": [],
        "left_predicted_demand": [],
        "right_predicted_demand": [],
        "left_days": [],
        "left_model_trend_days": [],
        "right_days": [],
        "right_model_trend_days": [],
        "left_non_null_model_demand_values_percent": [],
        "right_non_null_model_demand_values_percent": [],
        "method": [],
        "left_entropy": [],
        "right_entropy": [],
        "left_std": [],
        "right_std": [],
        "left_p_value": [],
        "right_p_value": [],
        "ate": [],
    }

    for (
        model_name,
        _,
        day_x_min_index,
        day_x_max_index,
        next_day_x_min_index,
        next_day_x_max_index,
        prediction_index,
        left_results,
        right_results,
    ) in rdd_both_ways_iterator(
        models=rdd_models,
        dict_mapping=dict_mapping,
        od_id=od_id,
        day_xs=day_xs,
        demand=demand,
        prices=prices,
    ):
        (
            left_demand_model,
            left_predicted_demand,
            left_fit_predictions,
            left_model_day_xs,
            left_prediction_price,
            left_model_demand,
            left_entropy,
            left_std,
            left_p_value,
            ate
        ) = left_results

        (
            right_demand_model,
            right_predicted_demand,
            right_fit_predictions,
            right_model_day_xs,
            right_prediction_price,
            right_model_demand,
            right_entropy,
            right_std,
            right_p_value,
            _
        ) = right_results

        left_model_day_xs = sorted(left_model_day_xs)
        right_model_day_xs = sorted(right_model_day_xs)
        # TODO check if wheould add +/-1
        prediction_day_x = day_xs[prediction_index]
        original_left_days = [day_x for day_x in original_day_xs if day_xs[day_x_min_index] <= day_x < prediction_day_x]
        original_right_days = [day_x for day_x in original_day_xs if prediction_day_x < day_x <= day_xs[next_day_x_max_index-1]]

        res_dict[time_column].append(day_xs[prediction_index])
        res_dict["left_price"].append(left_prediction_price)
        res_dict["right_price"].append(right_prediction_price)
        res_dict["left_predicted_demand"].append(left_predicted_demand)
        res_dict["right_predicted_demand"].append(right_predicted_demand)
        res_dict["left_days"].append(len(original_left_days))
        res_dict["left_model_trend_days"].append(len(left_model_day_xs))
        res_dict["right_days"].append(len(original_right_days))
        res_dict["right_model_trend_days"].append(len(right_model_day_xs))
        res_dict["left_non_null_model_demand_values_percent"].append((left_model_demand != 0).mean())
        res_dict["right_non_null_model_demand_values_percent"].append((right_model_demand != 0).mean())
        res_dict["method"].append(model_name)
        res_dict["left_entropy"].append(left_entropy)
        res_dict["right_entropy"].append(right_entropy)
        res_dict["left_std"].append(left_std)
        res_dict["right_std"].append(right_std)
        res_dict["left_p_value"].append(left_p_value)
        res_dict["right_p_value"].append(right_p_value)
        res_dict["ate"].append(ate)


    if evaluation_models:
        for (
            model_name,
            sequence_index,
            prediction_index,
            left_fit_predictions,
            left_prediction_price,
            left_model_day_xs,
            left_predicted_demand,
            right_fit_predictions,
            right_prediction_price,
            right_model_day_xs,
            right_predicted_demand,
            original_left_day_xs,
            original_right_day_xs,
        ) in evaluation_model_both_ways_iterator(
            models=evaluation_models,
            dict_mapping=dict_mapping,
            od_id=od_id,
            dict_mapping_df=dict_mapping_df,
            features_df=od_id_df.sort_values(time_column).reset_index(drop=True),
            data_price_column_name=price_column,
        ):
            
            prediction_day_x = day_xs[prediction_index]

            res_dict[time_column].append(day_xs[prediction_index])
            res_dict["left_price"].append(left_prediction_price)
            res_dict["right_price"].append(right_prediction_price)
            res_dict["left_predicted_demand"].append(left_predicted_demand)
            res_dict["right_predicted_demand"].append(right_predicted_demand)
            res_dict["left_days"].append(len(original_left_day_xs))
            res_dict["left_model_trend_days"].append(len(left_model_day_xs))
            res_dict["right_days"].append(len(original_right_day_xs))
            res_dict["right_model_trend_days"].append(len(right_model_day_xs))
            res_dict["left_non_null_model_demand_values_percent"].append((left_fit_predictions != 0).mean())
            res_dict["right_non_null_model_demand_values_percent"].append((right_fit_predictions != 0).mean())
            res_dict["method"].append(model_name)
            res_dict["left_entropy"].append(None)
            res_dict["right_entropy"].append(None)
            res_dict["left_std"].append(None)
            res_dict["right_std"].append(None)
            res_dict["left_p_value"].append(None)
            res_dict["right_p_value"].append(None)
    
    if tft_models:
        for (
            model_name,
            sequence_index,
            prediction_index,
            left_fit_predictions,
            left_prediction_price,
            left_model_day_xs,
            left_predicted_demand,
            right_fit_predictions,
            right_prediction_price,
            right_model_day_xs,
            right_predicted_demand,
            original_left_day_xs,
            original_right_day_xs,
        ) in tft_model_both_ways_iterator(
            models=tft_models,
            dict_mapping=dict_mapping,
            od_id=od_id,
            features_df=od_id_df.sort_values(time_column).reset_index(drop=True),
            dict_mapping_df=dict_mapping_df,
            price_column=price_column,
            day_init=day_init, # TODO change this in the name of the model
        ):
            
            prediction_day_x = day_xs[prediction_index]

            res_dict[time_column].append(day_xs[prediction_index])
            res_dict["left_price"].append(left_prediction_price)
            res_dict["right_price"].append(right_prediction_price)
            res_dict["left_predicted_demand"].append(left_predicted_demand)
            res_dict["right_predicted_demand"].append(right_predicted_demand)
            res_dict["left_days"].append(len(original_left_day_xs))
            res_dict["left_model_trend_days"].append(len(left_model_day_xs))
            res_dict["right_days"].append(len(original_right_day_xs))
            res_dict["right_model_trend_days"].append(len(right_model_day_xs))
            res_dict["left_non_null_model_demand_values_percent"].append((left_fit_predictions != 0).mean())
            res_dict["right_non_null_model_demand_values_percent"].append((right_fit_predictions != 0).mean())
            res_dict["method"].append(model_name)
            res_dict["left_entropy"].append(None)
            res_dict["right_entropy"].append(None)
            res_dict["left_std"].append(None)
            res_dict["right_std"].append(None)
            res_dict["left_p_value"].append(None)
            res_dict["right_p_value"].append(None)
    
    if tqdm_logger is not None:
        tqdm_logger.update(1)
    return pd.DataFrame(res_dict)


def _fit_rdd_model(
    model_name: str,
    model_fit_method: Callable,
    day_x_min_index: int,
    day_x_max_index: int,
    prediction_index: int,
    original_day_xs: np.ndarray,
    original_demand: np.ndarray,
    original_prices: np.ndarray,
    original_weekdays: np.ndarray,
    prediction_mode: Literal["next", "prev"],
) -> tuple[Callable, int, list, np.ndarray, Any, np.ndarray, float, float]:
    """
    Fit a simple model in the RDD process
    """
    model_demand = original_demand[day_x_min_index:day_x_max_index]
    model_day_xs = original_day_xs[day_x_min_index:day_x_max_index]
    model_prices = original_prices[day_x_min_index:day_x_max_index]
    model_weekdays = original_weekdays[day_x_min_index:day_x_max_index]

    prediction_weekday = original_weekdays[prediction_index]
    prediction_price = (
        original_prices[prediction_index] if prediction_mode == "prev" else original_prices[prediction_index - 1]
    )
    prediction_day_x = original_day_xs[prediction_index]

    sample_weights = None
    day_x_values_len = day_x_max_index - day_x_min_index

    if prediction_mode == "prev":
        model_day_xs = model_day_xs[::-1]
        model_demand = model_demand[::-1]
        model_weekdays = model_weekdays[::-1]
        model_prices = model_prices[::-1]

    if "weight" in model_name:
        day_x_limit = 14
        if day_x_values_len > day_x_limit:
            if "linear_weight" in model_name:
                sample_weights = np.arange(1, day_x_values_len + 1)
            else:
                sample_weights = np.exp(np.linspace(0, 5, day_x_values_len))
                sample_weights = sample_weights / sample_weights.max()  # type: ignore[union-attr]

    elif "limit" in model_name:
        day_x_limit = 8
        if day_x_values_len > day_x_limit:
            model_day_xs = model_day_xs[-day_x_limit:]
            model_demand = model_demand[-day_x_limit:]
            model_weekdays = model_weekdays[-day_x_limit:]
            model_prices = model_prices[-day_x_limit:]

    if "price" in model_name:
        # adding prices to features
        demand_model = model_fit_method(model_demand, model_day_xs, model_weekdays, model_prices, sample_weights)
        predicted_demand = demand_model(prediction_day_x, prediction_weekday, prediction_price)
        fit_predictions = [
            demand_model(day_x, weekday, price)
            for day_x, weekday, price in zip(model_day_xs, model_weekdays, model_prices)
        ]

    else:
        demand_model = model_fit_method(model_demand, model_day_xs, model_weekdays, sample_weights)
        predicted_demand = demand_model(prediction_day_x, prediction_weekday)
        fit_predictions = [demand_model(day_x, weekday) for day_x, weekday in zip(model_day_xs, model_weekdays)]

    if prediction_mode == "prev":
        fit_predictions = fit_predictions[::-1]
        model_day_xs = model_day_xs[::-1]
        model_demand = model_demand[::-1]

    return (
        demand_model,
        # max(predicted_demand, 0),
        predicted_demand,
        fit_predictions,
        model_day_xs,
        prediction_price,
        model_demand,
        entropy(model_demand),
        model_demand.std(),
    )


def _fit_both_ways_rdd_model(
    model_name: str,
    model_fit_method: Callable,
    day_x_min_index: int,
    day_x_max_index: int,
    next_day_x_min_index: int,
    next_day_x_max_index: int,
    prediction_index: int,
    original_day_xs: np.ndarray,
    original_demand: np.ndarray,
    original_prices: np.ndarray,
) -> tuple[Callable, int, list, np.ndarray, Any, np.ndarray, float, float, Any]:
    """
    Fit a simple model in the RDD process
    """
    model_demand = original_demand[day_x_min_index:next_day_x_max_index]
    model_day_xs = original_day_xs[day_x_min_index:next_day_x_max_index]

    prediction_day_x = original_day_xs[prediction_index]

    day_x_values_len = next_day_x_max_index - day_x_min_index
    sample_weights = None
    if "weight" in model_name:
        if day_x_values_len > 0:
            bandwith = 48
            if "linear_weight" in model_name:
                sample_weights = (np.abs(model_day_xs - prediction_day_x) <= bandwith).astype(float) * ( 1 - np.abs(model_day_xs - prediction_day_x) / bandwith)
            else:
                sample_weights = (np.abs(model_day_xs) <= bandwith).astype(float) * np.exp((- np.abs(model_day_xs) / (bandwith/5)))
                sample_weights = sample_weights / sample_weights.max()


    demand_model = model_fit_method(model_demand, model_day_xs, sample_weights, prediction_index-day_x_min_index)
    ate = None
    if isinstance(demand_model, tuple):
        demand_model, ate = demand_model
    
    epsilon = 1e-6
    left_fit_predictions = [demand_model(day_x) for day_x in original_day_xs[day_x_min_index:day_x_max_index]]
    right_fit_predictions = [demand_model(day_x) for day_x in original_day_xs[next_day_x_min_index:next_day_x_max_index]]
    right_predicted_demand = demand_model(prediction_day_x+epsilon)
    left_predicted_demand = demand_model(prediction_day_x-epsilon)

    # if sklearn_model is not None:

    
        

    # compute p value of predictions
    smf_model_str = "demand~day_x*threshold"
    p_value = smf.wls(
        smf_model_str, 
        pd.DataFrame({
            "day_x": np.concatenate((original_day_xs[day_x_min_index:day_x_max_index], original_day_xs[next_day_x_min_index:next_day_x_max_index]), axis=0) - prediction_day_x, 
            "demand": np.concatenate((original_demand[day_x_min_index:day_x_max_index], original_demand[next_day_x_min_index:next_day_x_max_index]), axis=0), 
            "threshold": np.concatenate((original_day_xs[day_x_min_index:day_x_max_index], original_day_xs[next_day_x_min_index:next_day_x_max_index]), axis=0) > prediction_day_x
        }).astype({"threshold": float})
    ).fit().pvalues["threshold"]


    # compute confidence based on bootstraping 
    model_day_xs = np.concatenate((original_day_xs[day_x_min_index:day_x_max_index], original_day_xs[next_day_x_min_index:next_day_x_max_index]), axis=0) - prediction_day_x
    model_demand = np.concatenate((original_demand[day_x_min_index:day_x_max_index], original_demand[next_day_x_min_index:next_day_x_max_index]), axis=0)
    sample_weights = np.concatenate((sample_weights[:(prediction_index-day_x_min_index)], sample_weights[(prediction_index-day_x_min_index+1):]), axis=0) if sample_weights is not None else None
    threshold = model_day_xs > 0


    if sample_weights is not None:
        non_null_weights_mask = sample_weights > 0
        model_day_xs = model_day_xs[non_null_weights_mask]
        model_demand = model_demand[non_null_weights_mask]
        sample_weights = sample_weights[non_null_weights_mask]
        threshold = threshold[non_null_weights_mask]


    def _fit_model_cb():
        indexes = np.random.choice(len(model_day_xs), len(model_day_xs), replace=True)
        kwargs = {}
        if sample_weights is not None:
            kwargs["weights"] = [sample_weights[idx] for idx in indexes]
        return smf.wls(
            smf_model_str, 
            pd.DataFrame({
                "day_x": [model_day_xs[idx] for idx in indexes], 
                "demand": [model_demand[idx] for idx in indexes], 
                "threshold": [threshold[idx] for idx in indexes],
            }).astype({"threshold": float}),
            **kwargs
        ).fit().params["threshold"]

    N = 50  # TODO change me after
    ates = []
    for _ in range(N):
        ates.append(_fit_model_cb())


    ate_std_results = np.std(ates)

    left_results = (
        demand_model,
        # max(left_predicted_demand, 0),
        left_predicted_demand,
        left_fit_predictions,
        original_day_xs[day_x_min_index:day_x_max_index],
        original_prices[day_x_min_index],
        original_demand[day_x_min_index:day_x_max_index],
        entropy(original_demand[day_x_min_index:day_x_max_index]),
        ate_std_results,
        # 0,
        p_value,
        # 0,
        ate,
    )

    right_results = (
        demand_model,
        # max(right_predicted_demand, 0),
        right_predicted_demand,
        right_fit_predictions,
        original_day_xs[next_day_x_min_index:next_day_x_max_index],
        original_prices[next_day_x_min_index],
        original_demand[next_day_x_min_index:next_day_x_max_index],
        entropy(original_demand[next_day_x_min_index:next_day_x_max_index]),
        ate_std_results,
        # 0,
        p_value,
        # 0,
        ate,
    )


    return (
        left_results, right_results
    )

def entropy(labels: np.ndarray, base: float = None) -> float:
    """Computes entropy of label distribution."""

    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.0

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)

    return ent



def compute_lgbm_rdd_values(
    lgbm_models: list[tuple[str, Any]],
    features_df: pd.DataFrame,
    price_column: str = "reference_price_standard_2",
    str_index: str = None,
    dict_mapping_query_str: str = None,
):

    def _groupby_cb(
        od_id_df: pd.DataFrame,
    ):
        

        od_id = od_id_df.od_id.values[0]
        res_dict = {
            "day_x": [],
            "left_price": [],
            "right_price": [],
            "left_predicted_demand": [],
            "right_predicted_demand": [],
            "method": [],
        }

        od_id_df = od_id_df.copy().sort_values("day_x").reset_index(drop=True)
        dict_mapping_df = od_id_df.copy()
        original_day_xs = od_id_df.day_x.values.tolist()
        if dict_mapping_query_str is not None:
            dict_mapping_df = dict_mapping_df.query(dict_mapping_query_str)

        od_id_df["reference_price"] = od_id_df["reference_price_standard_2"]
        day_x_values = od_id_df["day_x"].values

        for (
            model_name,
            sequence_index,
            prediction_index,
            left_fit_predictions,
            left_prediction_price,
            left_model_day_xs,
            left_predicted_demand,
            right_fit_predictions,
            right_prediction_price,
            right_model_day_xs,
            right_predicted_demand,
            original_left_days,
            original_right_days,
        ) in evaluation_model_both_ways_iterator(
            models=lgbm_models,
            dict_mapping=dict_mapping,
            od_id=od_id,
            features_df=od_id_df,
            dict_mapping_df=dict_mapping_df,
            model_price_column_name="reference_price"  # different from original price column
        ):
            res_dict["day_x"].append(day_x_values[prediction_index])
            res_dict["left_price"].append(left_prediction_price)
            res_dict["right_price"].append(right_prediction_price)
            res_dict["left_predicted_demand"].append(left_predicted_demand)
            res_dict["right_predicted_demand"].append(right_predicted_demand)
            res_dict["method"].append(model_name)
        return pd.DataFrame(res_dict)

    warnings.filterwarnings("ignore")
    sample_df = features_df.sort_values("day_x")

    dict_mapping: dict[str, Any] = {}

    dict_mapping_grouped_df = sample_df.groupby("od_id") if dict_mapping_query_str is None else sample_df.query(dict_mapping_query_str).groupby("od_id")
    tqdm.pandas(desc=f"od_id indexing: {str_index or 0}")
    _ = dict_mapping_grouped_df.progress_apply(update_mapping, dict_mapping=dict_mapping, price_column=price_column)

    tqdm.pandas(desc=f"Computing rdd per od_id for index {str_index or 0}")
    res_df = (
        sample_df.groupby("od_id").progress_apply(
            _groupby_cb,
        )
        .reset_index(level=1, drop=True)
        .reset_index()
    )

    res_df["delta_demand"] = res_df["left_predicted_demand"] - res_df["right_predicted_demand"]
    res_df["delta_price"] = res_df["left_price"] - res_df["right_price"]
    res_df["elasticity_value"] = (res_df["delta_demand"] / res_df["left_predicted_demand"].clip(0).replace(0, 1)) / (
        res_df["delta_price"] / res_df["left_price"]
    )
    res_df["absolute_elasticity_value"] = res_df["delta_demand"] * res_df["delta_price"] / res_df["delta_price"].abs()

    return res_df




def compute_rdd_values_n_jobs(
    df: pd.DataFrame,
    rdd_models: list[tuple[str, Callable]],
    n_jobs: int,
    od_id_column: str = "od_id",
    time_column: str = "day_x",
    static_columns: list[str] = None,
    price_column: str = "reference_price_standard_2",
    demand_column: str = "daily_demand_standard_2",
    evaluation_models: list[tuple[str, Any]] = None,
    day_init: int = 20,
    tft_models: list[tuple[str, Any]] = None,
    dict_mapping_query_str: str = None,
    min_delta_price: int = 3,
):
    od_ids = df[od_id_column].unique().tolist()
    r = len(od_ids) % n_jobs

    chunk_size = len(od_ids) // n_jobs
    od_ids_chunks = [
        od_ids[(i*chunk_size):(((i+1)*chunk_size))]
        for i in range(n_jobs)
    ]
    for i in range(r):
        od_ids_chunks[i].append(od_ids[-i-1])


 
    print("Running parallel jobs after chunks")

    results = joblib.Parallel(n_jobs=n_jobs, backend='loky')(
    joblib.delayed(
        compute_rdd_values_n_jobs_cb)(
            df=df.query(f"{od_id_column} in {od_ids_chunk}"),
            rdd_models=rdd_models,
            static_columns=static_columns,
            price_column=price_column,
            demand_column=demand_column,
            str_index=str(idx),
            evaluation_models=evaluation_models,
            day_init=day_init,
            tft_models=tft_models,
            dict_mapping_query_str=dict_mapping_query_str,
            od_id_column=od_id_column,
            time_column=time_column,
            min_delta_price=min_delta_price,
        ) 
        for idx, od_ids_chunk in tqdm(enumerate(od_ids_chunks), desc="Computing arguments dataframe", total=len(od_ids_chunks))
    )

    print("Contaning results")
    return pd.concat(results, axis=0)

def compute_rdd_values_n_jobs_cb(
    df: pd.DataFrame,
    rdd_models: list[tuple[str, Callable]],
    min_delta_price: int,
    od_id_column: str = "od_id",
    time_column: str = "day_x",
    static_columns: list[str] = None,
    price_column: str = "reference_price_standard_2",
    demand_column: str = "daily_demand_standard_2",
    str_index: str = None,
    evaluation_models: list[tuple[str, Any]] = None,
    day_init: int = 20,
    tft_models: list[tuple[str, Any]] = None,
    dict_mapping_query_str: str = None,
):
    
    warnings.filterwarnings("ignore")
    # parsed_models = []
    # if isinstance(rdd_models, dict):
    #     for model_name, (model_fn, model_fn_kwargs) in rdd_models.items():
    #         if isinstance(model_fn, str):
    #             model_fn = from_fully_qualified_import(model_fn)
    #             if model_fn_kwargs is not None and len(model_fn_kwargs) > 0:
    #                 model_fn = model_fn(**model_fn_kwargs)  # type: ignore[operator]
    #         parsed_models.append((model_name, model_fn))
    # else:
    #     parsed_models = rdd_models  # type: ignore
    parsed_models = rdd_models

    sample_df = df.sort_values(time_column)

    dict_mapping: dict[str, Any] = {}

    dict_mapping_grouped_df = sample_df.groupby(od_id_column) if dict_mapping_query_str is None else sample_df.query(dict_mapping_query_str).groupby(od_id_column)
    tqdm.pandas(desc=f"od_id indexing: {str_index or 0}")
    _ = dict_mapping_grouped_df.progress_apply(
        update_mapping, 
        dict_mapping=dict_mapping, 
        price_column=price_column, 
        od_id_column=od_id_column, 
        time_column=time_column,
        min_delta_price=min_delta_price
    )


    # tqdm_out = TqdmToLogger(get_app().logger, level=logging.INFO)
    # tqdm_logger = tqdm(file=tqdm_out, total=len(grouped_df), mininterval=10, desc=f"Computing rdd per od_id for index {str_index or 0}")

    tqdm.pandas(desc=f"Computing rdd per od_id for index {str_index or 0}")
    res_df = (
        sample_df.groupby(od_id_column).progress_apply(
            _compute_od_id_rdd_values,
            rdd_models=parsed_models,
            demand_column=demand_column,
            price_column=price_column,
            dict_mapping=dict_mapping,
            filter_warning=True,
            # tqdm_logger=tqdm_logger,
            evaluation_models=evaluation_models,
            day_init=day_init,
            tft_models=tft_models,
            dict_mapping_query_str=dict_mapping_query_str,
            time_column=time_column,
            od_id_column=od_id_column,
        )
        .reset_index(level=1, drop=True)
        .reset_index()
    )

    res_df["delta_demand"] = res_df["right_predicted_demand"] - res_df["left_predicted_demand"]
    res_df["delta_price"] = res_df["right_price"] - res_df["left_price"]
    res_df["elasticity_value"] = (res_df["delta_demand"] / res_df["left_predicted_demand"].clip(0).replace(0, 1)) / (
        res_df["delta_price"] / res_df["left_price"]
    )
    res_df["absolute_elasticity_value"] = res_df["delta_demand"] * res_df["delta_price"] / res_df["delta_price"].abs()

    if static_columns is None:
        return res_df

    return res_df.astype({"day_x": df.day_x.dtype}).join(
        df[["od_id"] + static_columns].drop_duplicates("od_id").set_index(["od_id"]), on=["od_id"], how="left"
    )

