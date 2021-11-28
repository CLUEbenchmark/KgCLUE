from transformers.data.metrics.squad_metrics import squad_evaluate

results_default_thresh = squad_evaluate(examples,
                                        preds,
                                        no_answer_probs=null_odds,
                                        no_answer_probability_threshold=1.0)
