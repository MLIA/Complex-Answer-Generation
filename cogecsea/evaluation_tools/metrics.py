import sys
import datasets

current_module = sys.modules[__name__]

def rouge(generated, reference, **kwargs):
    metric = datasets.load_metric('rouge')
    metric.add_batch(predictions=generated, references=reference)
    results = metric.compute()
    return results

def bleu(generated, reference, **kwargs):
    metric = datasets.load_metric('sacrebleu')
    metric.add_batch(predictions=generated, references=[[r] for r in reference])
    results = metric.compute()
    return results


def bert_score(generated, reference, lang="en", **kwargs):
    metric = datasets.load_metric("bertscore")
    metric.add_batch(predictions=generated, references=reference)
    results = metric.compute(lang=lang, verbose=True)
    return results

def meteor(generated, reference, **kwargs):
    metric = datasets.load_metric("meteor")
    metric.add_batch(predictions=generated, references=reference)
    results = metric.compute()
    return results

def evaluate_metrics(generated, references, *metrics, **kwargs):
    results = {
               metric: getattr(current_module, metric, kwargs)(generated, references) 
               for metric in metrics
    }

    return results